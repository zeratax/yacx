public class Example{

    public static void main(String[] args){
        Executor.init();

        //Testdata
        float n = 128*32;
        float a = 5.1f;
        float[] x = new float[n];
        float[] y = new float[n];
        for (size_t i = 0; i < n; ++i) {
            x[i] = i;
            y[i] = 2*i;
        }

        //Initialize Arguments
        KernelArg aArg, xArg, yArg, outArg, nArg;
        aArg = KernelArg.create(new float[]{a}, false);
        xArg = KernelArg.create(x, false);
        yArg = KernelArg.create(y, false);
        outArg = KernelArg.create(n*4);
        nArg = KernelArg.create(new float[]{a}, false);

        //Create Program
        Program saxpyKernel = Program.create(loadFile("kernels/saxpy"), "saxpy", "");

        //Run Program
        double runtime = Executor.execute(saxpyKernel, new KernelArg[]{aArg, xArg, yArg, outArg, nArg});

        //Get Result
    }

    private static String loadFile(String filename){
        return new String(Files.readAllBytes(new File(kernelName).toPath()));
    }
}