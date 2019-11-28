import scala.io.Source

object Utils {
    def loadFile(fileName : String) : String = {
        val lines = Source.fromFile(fileName).getLines.mkString("\n")
        return lines
    }
}
