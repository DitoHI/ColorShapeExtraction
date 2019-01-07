import java.io.File
import java.io.IOException
import java.nio.file.Paths
import javax.imageio.ImageIO

fun main(args: Array<String>) {
    val currentRelativePath = Paths.get("")
    val s = currentRelativePath.toAbsolutePath().toString()
    val mainDir = "$s/src/images/ex1.JPG"
    val file = File(mainDir)

    if (file.isFile) {
        try {
            val image = ImageIO.read(file)
            // Color Extraction
            val colorExtraction = ColorExtraction(image)
            println("ColorFeature_Mean[Color Read: ${colorExtraction.meanRed}, Color Green: ${colorExtraction.meanGreen}, Color Blue: ${colorExtraction.meanBlue}]")
            println("ColorFeature_StandardDeviation[Color Read: ${colorExtraction.stdRed}, Color Green: ${colorExtraction.stdGreen}, Color Blue: ${colorExtraction.stdBlue}]")

            println("")

            // Feature Extraction
            val shapeExtraction = ShapeExtraction(image)
            println("ShapeFeature[Area: ${shapeExtraction.area}, Perimeter: ${shapeExtraction.perimeter}, Circularity: ${shapeExtraction.circulatiry}]")
        } catch (e: IOException) {
            println("Error ${e.message}")
        }
    }
}