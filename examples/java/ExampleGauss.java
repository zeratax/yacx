import java.io.File;
import java.nio.file.Files;
import java.util.Arrays;

public class ExampleGauss {
//    void writePPM(Pixel *pixels, String filename, int width, int height) {
//        std::ofstream outputFile(filename, std::ios::binary);
//      
//        // write header:
//        outputFile << "P6\n" << width << " " << height << "\n255\n";
//      
//        outputFile.write(reinterpret_cast<const char *>(pixels),
//                         sizeof(Pixel) * width * height);
//    }
//      
      // Pointer returned must be explicitly freed!
      PixelArray readPPM(String filename, int width, int height) {
        
      
        // parse harder
        // first line: P6\n
        inputFile.ignore(2, '\n'); // ignore P6
        // possible comments:
        while (inputFile.peek() == '#') {
          inputFile.ignore(1024, '\n');
        } // skip comment
        // next line: width_height\n
        inputFile >> (*width);
        inputFile.ignore(1, ' '); // ignore space
        inputFile >> (*height);
        inputFile.ignore(1, '\n'); // ignore newline
        // possible comments:
        while (inputFile.peek() == '#') {
          inputFile.ignore(1024, '\n');
        } // skip comment
        // last header line: 255\n:
        inputFile.ignore(3, '\n'); // ignore 255 and newline
      
        Pixel *data = new Pixel[(*width) * (*height)];
        
        inputFile.read(reinterpret_cast<char *>(data),
                       sizeof(Pixel) * (*width) * (*height));
      
        return data;
      }

	static float[][] calculateWeights(int size) {
		float[][] weights = new float[size][size];
		
        float sigma = 1.0f;
        float r, s = 2.0f * sigma * sigma;
      
        // sum is for normalization
        float sum = 0.0f;
      
        // generate weights for 5x5 kernel
        for (int x = -2; x <= 2; x++) {
          for (int y = -2; y <= 2; y++) {
            r = x * x + y * y;
            weights[x + 2][y + 2] = (float) (Math.exp(-(r / s)) / (Math.PI * s));
            sum += weights[x + 2][y + 2];
          }
        }
      
        // normalize the weights
        for (int i = 0; i < 5; ++i) {
          for (int j = 0; j < 5; ++j) {
            weights[i][j] /= sum;
          }
        }
        
        return weights;
    }

	public static void main(String[] args) {
		// Load Libary
		Executor.loadLibary();

		// Testdata
		final String inputFile = "lenna.ppm";
		final String outputFile = "output.ppm";
		
		float[][] weigths = calculateWeights(5);
		int width;
		int height;
		
		PixelArray image = 

		// Initialize Arguments
		KernelArg aArg, nArg, xArg, yArg;
		FloatArg outArg;
		aArg = FloatArg.createValue(a);
		xArg = FloatArg.create(x);
		yArg = FloatArg.create(y);
		outArg = FloatArg.createOutput(n);
		nArg = IntArg.createValue(n);

		// Create Program
		String kernelString = Utils.loadFile("saxpy.cu");
		Program saxpy = Program.create(kernelString, "saxpy");

		// Create compiled Kernel
		Kernel saxpyKernel = saxpy.compile();

		// Compile and launch Kernel
		KernelTime executionTime = saxpyKernel.launch(numThreads, numBlocks, aArg, xArg, yArg, outArg, nArg);

		// Get Result
		float[] out = outArg.asFloatArray();

		// Print Result
		System.out.println("\nsaxpy-Kernel sucessfully launched:");
		System.out.println(executionTime);
		System.out.println("\nInput a: " + a);
		System.out.println("Input x: " + Arrays.toString(x));
		System.out.println("Input y: " + Arrays.toString(y));
		System.out.println("Result:  " + Arrays.toString(out));
	}
	
	static class PixelArray {
		private byte[] pixel;
		
		public PixelArray(int length) {
			pixel = new byte[length*3];
		}
		
		public void setPixel(int i, byte r, byte b, byte g) {
			int j = i*3;
			pixel[j] = r;
			pixel[j+1] = b;
			pixel[j+2] = g;
		}
		
		public byte[] asByteArray() {
			return pixel;
		}
	}
	
	
	
	
}

