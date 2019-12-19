object ExampleGauss {
    def writePPM(Pixel *pixels, const char *filename, int width, int height) : Unit {
        std::ofstream outputFile(filename, std::ios::binary);
      
        // write header:
        outputFile << "P6\n" << width << " " << height << "\n255\n";
      
        outputFile.write(reinterpret_cast<const char *>(pixels),
                         sizeof(Pixel) * width * height);
    }
      
      // Pointer returned must be explicitly freed!
      Pixel *readPPM(const char *filename, int *width, int *height) {
        std::ifstream inputFile(filename, std::ios::binary);
      
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
      
      void calculateWeights(float weights[5][5]) {
        float sigma = 1.0;
        float r, s = 2.0 * sigma * sigma;
      
        // sum is for normalization
        float sum = 0.0;
      
        // generate weights for 5x5 kernel
        for (int x = -2; x <= 2; x++) {
          for (int y = -2; y <= 2; y++) {
            r = x * x + y * y;
            weights[x + 2][y + 2] = exp(-(r / s)) / (M_PI * s);
            sum += weights[x + 2][y + 2];
          }
        }
      
        // normalize the weights
        for (int i = 0; i < 5; ++i) {
          for (int j = 0; j < 5; ++j) {
            weights[i][j] /= sum;
          }
        }
      }

    def main(args: Array[String]) : Unit = {
        //Load Libary
        Executor.loadLibary()

        //Testdata
        val numThreads = 16
        val numBlocks = 1

        val n = 16
        val in = new Array[Int](n)

        for (i <- 0 until n){
            in(i) = i
        }

        //Initialize Arguments
        val srcArg = IntArg.create(in: _*)
        val outArg = IntArg.createOutput(n/2)
        val counterArg = IntArg.create(Array[Int](0), true)
        val nArg = IntArg.createValue(n)

        //Create Program
        val kernelString = Utils.loadFile("filter_k.cu")
        val filter = Program.create(kernelString, "filter_k")

        //Create compiled Kernel
        val filterKernel = filter.compile()

        //Launch Kernel
        val executionTime = filterKernel.launch(numThreads, numBlocks, outArg, counterArg, srcArg, nArg)

        //Get Result
        val out = outArg.asIntArray()
        val counter = counterArg.asIntArray()(0)

        //Print Result
        println("\nfilter-Kernel sucessfully launched:");
        println(executionTime);
        println("\nInput:          [" + in.mkString(", ") + "]");
        println("Result counter: " + counter);
        println("Result:         [" + out.mkString(", ") + "]");
  }
}