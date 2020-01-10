import java.io.IOException;

import yacx.Executor;
import yacx.IntArg;
import yacx.Kernel;
import yacx.KernelTime;
import yacx.Program;
import yacx.Utils;

public class ExampleTemplateKernel {
	public static void main(String[] args) throws IOException {
		//Load Libary
        Executor.loadLibary();
        
        //Testdata
        final int numThreads = 1;
        final int numBlocks = 1;

        //Initialize Arguments
        IntArg resultArg = IntArg.createOutput(1);

        //Create Program
        String templateKernelString = Utils.loadFile("kernels/template.cu");
        Program f3 = Program.create(templateKernelString, "f3");
        
        //Setting template-Parameters
        final String template = "int";
        f3.instantiate(template);

        //Create compiled Kernel
        Kernel f3Kernel = f3.compile();

        //Compile and launch Kernel
        KernelTime executionTime = f3Kernel.launch(numThreads, numBlocks, resultArg);

        //Get Result
        int result = resultArg.asIntArray()[0];

        //Print Result
        System.out.println("\ntemplateKernel-Kernel sucessfully launched:");
        System.out.println(executionTime);
        System.out.println("\nTemplate: " + template);
        System.out.println("Result:    " + result);
	}
}
