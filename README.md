# Color and Shape Feature Extraction

For starter, this library helps you to get the color and shape feature in your image. See our class `Main.kt` to helps you understand the basic of usage

### Initialize the Image
First you must initialize the image from your directory. The initialized image will be `BufferedImage` type.
 
```kotlin
val currentRelativePath = Paths.get("")
val s = currentRelativePath.toAbsolutePath().toString()
val mainDir = "$s/images/ex1.JPG"
val file = File(mainDir)
val image = ImageIO.read(file)
```

### Color Extraction

This class is used to get the color feature from an image. 

```kotlin
val colorExtraction = ColorExtraction(image)
println("ColorFeature_Mean[Color Read: ${colorExtraction.meanRed}, Color Green: ${colorExtraction.meanGreen}, Color Blue: ${colorExtraction.meanBlue}]")
println("ColorFeature_StandardDeviation[Color Read: ${colorExtraction.stdRed}, Color Green: ${colorExtraction.stdGreen}, Color Blue: ${colorExtraction.stdBlue}]")
// this is the output by using the example image
// ColorFeature_Mean[Color Read: 137.90780639648438, Color Green: 145.11952209472656, Color Blue: 122.38722229003906]
// ColorFeature_StandardDeviation[Color Read: 73.14114477792411, Color Green: 67.07990598971824, Color Blue: 80.31961305553705]
```

### Shape Extraction

This class is used to get the shape feature from an image. 

```kotlin
val shapeExtraction = ShapeExtraction(image)
            println("ShapeFeature[Area: ${shapeExtraction.area}, Perimeter: ${shapeExtraction.perimeter}, Circularity: ${shapeExtraction.circulatiry}]")
// this is the output by using the example image
// ShapeFeature[Area: 863.0, Perimeter: 2388.59117269516, Circularity: 0.0019008025843089407]
```
