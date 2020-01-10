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

public class ExampleGauss {
	static void writePPM(PixelArray pixels, String filename) throws IOException {
		try (FileOutputStream os = new FileOutputStream(new File(filename));
				BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(os))) {
			
			//Writer Header
			writer.write("P6\n");
			
			//Write width and height
			writer.write("" + pixels.getWidth() + " " + pixels.getHeight() + "\n");
			
			//Write header-end
			writer.write("255\n");
			
			writer.flush();
			
			//Writer Pixels
			os.write(pixels.getPixel());
		}
	}
	
    static PixelArray readPPM(String filename) throws IOException {
    	File file = new File(filename);
    	
    	try (InputStream is = new FileInputStream(file);
          	  BufferedReader reader = new BufferedReader(new FileReader(file))) {
    		
            //Skip first 3 Bytes (Header)
            reader.skip(3);
            
            //Ignore comments
            String line;
            while ((line = reader.readLine()).startsWith("#"));
            
            //Read width and height
            int width = Integer.parseInt(line.substring(0, line.indexOf(" ")));
            int height = Integer.parseInt(line.substring(line.indexOf(" ")+1));
            
            //Ignore comments
            while ((line = reader.readLine()).startsWith("#"));
            
            //Last header line
            assert(line.equals("255"));
            
            //Read pixels
            PixelArray pixels = new PixelArray(width, height);
            
            is.skip(file.length()-pixels.getPixel().length);
            
            is.read(pixels.getPixel());
            
            return pixels;
    	}
    }

	static float[] calculateWeights(int size) {
		float[] weights = new float[size*size];
		
        float sigma = 1.0f;
        float r, s = 2.0f * sigma * sigma;
      
        // sum is for normalization
        float sum = 0.0f;
      
        // generate weights for 5x5 kernel
        for (int x = -size/2; x <= size/2; x++) {
          for (int y = -size/2; y <= size/2; y++) {
            r = x * x + y * y;
            weights[(x + 2)*size + y + 2] = (float) (Math.exp(-(r / s)) / (Math.PI * s));
            sum += weights[(x + 2)*size + y + 2];
          }
        }
      
        // normalize the weights
        for (int i = 0; i < size; ++i) {
          for (int j = 0; j < size; ++j) {
            weights[i*size + j] /= sum;
          }
        }
        
        return weights;
    }

	public static void main(String[] args) throws IOException {
		// Load Libary
		Executor.loadLibary();

		// Testdata
		final String inputFile = "kernels/lena.ppm";
		final String outputFile = "output.ppm";
		
		float[] weigths = calculateWeights(5);
		
		PixelArray image = readPPM(inputFile);
		int width = image.getWidth();
		int height = image.getHeight();

		// Initialize Arguments
		KernelArg weightsArg, widthArg, heightArg;
		ByteArg imageArg;
		weightsArg = FloatArg.create(weigths);
		widthArg = IntArg.createValue(width);
		heightArg = IntArg.createValue(height);
		imageArg = ByteArg.create(image.getPixel(), true);

		//Create Program
		String kernelString = Utils.loadFile("gauss.cu");

		//Compile and launch Kernel
		KernelTime executionTime = Executor.launch(kernelString, "gaussFilterKernel", width, height, 1,
				1, 1, 1, imageArg, weightsArg, widthArg, heightArg);

		//Get Result
		PixelArray imageFiltered = new PixelArray(width, height, imageArg.asByteArray());
		
		System.out.println("\ngauss-Kernel sucessfully launched:");
        System.out.println(executionTime);
        System.out.println("\nInputfile: " + inputFile);
        System.out.println("Outputfile: " + outputFile);

		//Write Result
		writePPM(imageFiltered, outputFile);
	}
	
	static class PixelArray {
		private final byte[] pixel;
		private final int width;
		private final int height;
		
		public PixelArray(int width, int height) {
			this.width = width;
			this.height = height;
			
			pixel = new byte[width*height*3];
		}
		
		public PixelArray(int width, int height, byte[] pixel) {
			assert(pixel.length == width*height*3);
			
			this.width = width;
			this.height = height;
			this.pixel = pixel;
		}
		
		public byte[] getPixel() {
			return pixel;
		}

		public int getWidth() {
			return width;
		}

		public int getHeight() {
			return height;
		}
	}
}
