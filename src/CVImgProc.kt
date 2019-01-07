import org.apache.commons.math3.stat.descriptive.moment.Mean
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation
import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.awt.Color
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

open class CVImgProc {
    lateinit var image: BufferedImage
    var width = 0
    var height = 0
    var totalPixel = 0
    constructor()
    constructor(_image: BufferedImage) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
        image = _image
        width = _image.width
        height = _image.height
        totalPixel = width * height
    }

    fun bufferedImageToMat(_image: BufferedImage): Mat {
        val mat = Mat(_image.height, _image.width, CvType.CV_8UC3)
        val data = (_image.raster.dataBuffer as DataBufferByte).data
        mat.put(0, 0, data)
        return mat
    }

    fun MatToBufferedImage(_matrix: Mat): BufferedImage {
        val mob = MatOfByte()
        Imgcodecs.imencode(".jpg", _matrix, mob)
        val ba = mob.toArray()
        return ImageIO.read(ByteArrayInputStream(ba))
    }

    fun watershedSegmentation(_image: BufferedImage): BufferedImage {
        val srcMat = this.bufferedImageToMat(_image)
        val grayMat = srcMat
        val rgba = Mat()
        val threeChannel = Mat()

        Imgproc.cvtColor(grayMat, rgba, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgba, threeChannel, Imgproc.COLOR_RGB2GRAY)
        Imgproc.threshold(threeChannel, threeChannel, 100.0, 255.0, Imgproc.THRESH_BINARY)

        val fg = Mat(rgba.size(), CvType.CV_8U)
        Imgproc.erode(threeChannel, fg, Mat(), Point(-1.0, -1.0), 2)
        val bg = Mat(rgba.size(), CvType.CV_8U)
        Imgproc.dilate(threeChannel, bg, Mat(), Point(-1.0, -1.0), 3)
        Imgproc.threshold(bg, bg, 1.0, 128.0, Imgproc.THRESH_BINARY)
        val markers = Mat(rgba.size(), CvType.CV_8U, Scalar(0.0))
        Core.add(fg, bg, markers)

        val markerTempo = Mat()
        markers.convertTo(markerTempo, CvType.CV_32S)
        Imgproc.watershed(rgba, markerTempo)
        markerTempo.convertTo(markers, CvType.CV_8U)

        Imgproc.applyColorMap(markers, markers, 4)
        // convert Mat fg to BufferedImage
        val segmentedImage = this.MatToBufferedImage(fg)

        return segmentedImage
    }
}

class ColorExtraction: CVImgProc {
    lateinit var segmentedBufferedImage: BufferedImage
    var meanRed: Double = 0.0
    var meanGreen: Double = 0.0
    var meanBlue: Double = 0.0
    var stdRed: Double = 0.0
    var stdGreen: Double = 0.0
    var stdBlue: Double = 0.0
    constructor()
    constructor(_image: BufferedImage): super(_image) {
        this.segmentedBufferedImage = super.watershedSegmentation(_image)
        this.initalizePixel(totalPixel)
    }

    private fun initalizePixel(_totalPixel: Int) {
        var sumRed = 0.0
        var sumGreen = 0.0
        var sumBlue = 0.0
        val redArray = DoubleArray(_totalPixel)
        val blueArray = DoubleArray(_totalPixel)
        val greenArray = DoubleArray(_totalPixel)
        var start = 0
        for (i in 0 until (this.segmentedBufferedImage).width) {
            for (j in 0 until (this.segmentedBufferedImage).height) {
                val pixel = (this.segmentedBufferedImage).getRGB(i, j)
                val originalPixel = image.getRGB(i, j)
                var newPixel = 0
                if (pixel.equals(-1)) {
                    newPixel = pixel
                } else {
                    newPixel = originalPixel
                }
                val red = Color(newPixel).red.toDouble()
                val blue = Color(newPixel).blue.toDouble()
                val green = Color(newPixel).green.toDouble()
                sumRed+= red
                sumBlue+= blue
                sumGreen+= green

                redArray[start] = red
                blueArray[start] = blue
                greenArray[start] = green
                ++start
            }
        }
        // calculate mean and standard deviation
        val mean = Mean()
        this.meanRed  = mean.evaluate(redArray)
        this.meanGreen = mean.evaluate(greenArray)
        this.meanBlue= mean.evaluate(blueArray)
        val std = StandardDeviation(false)
        this.stdRed = std.evaluate(redArray)
        this.stdGreen = std.evaluate(greenArray)
        this.stdBlue = std.evaluate(blueArray)
    }
}

class ShapeExtraction: CVImgProc {
    var area: Double = 0.0
    var perimeter: Double = 0.0
    var circulatiry: Double = 0.0
    lateinit var cannyImage: BufferedImage
    constructor()
    constructor(_image: BufferedImage): super(_image) {
        // get contours
        val originalMat = bufferedImageToMat(image)
        val grayMat = Mat()
        val cannyEdges = Mat()
        val hierarchy = Mat()
        val contourList = ArrayList<MatOfPoint>()

        Imgproc.cvtColor(originalMat, grayMat, Imgproc.COLOR_BGR2GRAY)
        Imgproc.Canny(originalMat, cannyEdges, 10.0, 100.0)
        Imgproc.findContours(cannyEdges, contourList, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

        val contours = Mat()
        contours.create(cannyEdges.rows(), cannyEdges.cols(), CvType.CV_8UC3)
        for (i in 0 until contourList.size) {
            Imgproc.drawContours(contours, contourList, i, Scalar(255.0, 255.0, 255.0), -1)
        }
        this.cannyImage = super.MatToBufferedImage(contours)
        this.setShapeFeature(contourList)
    }

    private fun setShapeFeature(contourList: ArrayList<MatOfPoint>) {
        var currentMax = 0.0
        var arcLength = 0.0
        var circularity = 0.0
        val perimeter = MatOfPoint2f()
        for (c in contourList) {
            val area = Imgproc.contourArea(c)
            if (area > currentMax) {
                currentMax = area
                c.convertTo(perimeter, CvType.CV_32FC2)
                arcLength = Imgproc.arcLength(perimeter, true)
                if (arcLength > 0) {
                    circularity = 4 * Math.PI * area / Math.pow(arcLength, 2.0)
                }
            }
        }
        this.area = currentMax
        this.perimeter = arcLength
        this.circulatiry = circularity
    }
}