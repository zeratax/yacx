import scala.io.Source

object Utils {
    def loadFile(fileName : String) : String = {
        val lines = Source.fromFile("../examples/kernels/" + fileName).getLines.mkString("\n")
        return lines
    }
}
