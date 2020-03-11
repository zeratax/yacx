import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;

import yacx.ByteArg;
import yacx.Executor;
import yacx.FloatArg;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.KernelTime;
import yacx.Utils;

object ExampleGauss {
  	def writePPM(pixels: PixelArray, filename: String) : Unit = {
    		var os : FileOutputStream = null
    		var writer : BufferedWriter = null
    				
        try {
    			os = new FileOutputStream(new File(filename))
    		  writer = new BufferedWriter(new OutputStreamWriter(os))
    		  
    			//Writer Header
    			writer.write("P6\n")
    			
    			//Write width and height
    			writer.write("" + pixels.getWidth() + " " + pixels.getHeight() + "\n")
    			
    			//Write header-end
    			writer.write("255\n")
    			
    			writer.flush()
    			
    			//Writer Pixels
    			os.write(pixels.getPixel())
    		} finally {
    	    if (os != null)
    	        os.close()
    	    if (writer != null)
    	        writer.close()
    	}
  	}
	
    def readPPM(filename: String) : PixelArray = {
        val file = new File(filename)
        var is : FileInputStream = null
        var reader : BufferedReader = null
        
        try {
            is = new FileInputStream(file)
            reader = new BufferedReader(new FileReader(file))
              
            //Skip first 3 Bytes (Header)
            reader.skip(3)
            
            //Ignore comments
            var line : String = null
            do {
                line = reader.readLine()
            } while (line.startsWith("#"));
            
            //Read width and height
            val width = Integer.parseInt(line.substring(0, line.indexOf(" ")))
            val height = Integer.parseInt(line.substring(line.indexOf(" ")+1))
            
            //Ignore comments
            do {
                line = reader.readLine()
            } while (line.startsWith("#"));
            
            //Last header line
            assert(line.equals("255"))
            
            //Read pixels
            val pixels = new PixelArray(width, height)
            
            is.skip(file.length()-pixels.getPixel().length);
            
            is.read(pixels.getPixel())
            
            return pixels
    	} finally {
    	    if (is != null)
    	        is.close()
    	    if (reader != null)
    	        reader.close()
    	}
    }

  	def calculateWeights(size: Int) : Array[Float] = {
  		  val weights = new Array[Float](size*size)
  		
        val sigma = 1.0f
        var r, s = 2.0f * sigma * sigma
        
        // sum is for normalization
        var sum = 0.0f
        
        // generate weights for 5x5 kernel
        for (x <- -size/2 until size/2 + 1) {
          for (y <- -size/2 until size/2 + 1) {
            r = x * x + y * y
            weights((x + 2)*size + y + 2) = ((Math.exp(-(r / s)) / (Math.PI * s))).toFloat
            sum += weights((x + 2)*size + y + 2)
          }
        }
        
        // normalize the weights
        for (i <- 0 until size) {
          for (j <- 0 until size) {
            weights(i*size + j) /= sum
          }
        }
          
        return weights
    }

  	def main(args: Array[String]) : Unit = {
    		// Load Libary
    		Executor.loadLibary()
    
    		// Testdata
    		val inputFile = "kernels/lena.ppm"
    		val outputFile = "output.ppm"
    		
    		val weigths = calculateWeights(5)
    		
    		val image = readPPM(inputFile)
    		val width = image.getWidth()
    		val height = image.getHeight()
    
    		// Initialize Arguments
    		val weightsArg = FloatArg.create(weigths: _*)
    		val widthArg = IntArg.createValue(width)
    		val heightArg = IntArg.createValue(height)
    		val imageArg = ByteArg.create(image.getPixel(), true)
    
    		//Create Program
    		val kernelString = Utils.loadFile("kernels/gauss.cu")
    
    		//Compile and launch Kernel
    		val executionTime = Executor.launch(kernelString, "gaussFilterKernel", width, height, 1,
    				1, 1, 1, imageArg, weightsArg, widthArg, heightArg)
    
    		//Get Result
    		val imageFiltered = new PixelArray(width, height, imageArg.asByteArray())
    		
    		println("\ngauss-Kernel sucessfully launched:")
        println(executionTime)
        println("\nInputfile: " + inputFile)
        println("Outputfile: " + outputFile)
    
    		//Write Result
    		writePPM(imageFiltered, outputFile)
  	}
  	
  	class PixelArray(width: Int, height: Int, pixel: Array[Byte]) {
  	    
  	    def this(width: Int, height: Int) {
  	      this(width, height, new Array[Byte](width*height*3))
  	    }
  	    
    		def getPixel() : Array[Byte] = {
    		    return pixel
    		}
    
    		def getWidth() : Int = {
    			  return width
    		}
    
    		def getHeight() : Int = {
    			  return height
    		}
    }
} 	
 