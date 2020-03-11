import yacx.Executor;
import yacx.IntArg;
import yacx.Kernel;
import yacx.KernelTime;
import yacx.Program;
import yacx.Utils;

object ExampleTemplateKernel {
    def main(args: Array[String]) : Unit = {
        //Load Libary
        Executor.loadLibary()

        //Testdata
        val numThreads = 1
        val numBlocks = 1

        //Initialize Arguments
        val resultArg = IntArg.createOutput(1)

        //Create Program
        val templateKernelString = Utils.loadFile("kernels/template.cu")
        val f3 = Program.create(templateKernelString, "f3")
        
        //Setting template-Parameters
        val template = "int";
        f3.instantiate(template);

        //Create compiled Kernel
        val f3Kernel = f3.compile()

        //Launch Kernel
        val executionTime = f3Kernel.launch(numThreads, numBlocks, resultArg)

        //Get Result
        val result = resultArg.asIntArray()(0)

        //Print Result
        println("\ntemplateKernel-Kernel sucessfully launched:")
        println(executionTime)
        println("\nTemplate: " + template)
        println("Result:    " + result)
    }
}
