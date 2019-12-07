import java.io.IOException;
import java.util.Arrays;

public class ExampleFilterExecutor {
    public static void main(String[] args) throws IOException {
        //Load Libary
        Executor.loadLibary();

        //Create OutputArgument
        int n = 4;
        IntArg out = IntArg.createOutput(n/2);

        //Compile and launch Kernel
        System.out.println(Executor.launch("filter_k", n, 1, out, IntArg.createOutput(1),
                                                 IntArg.create(0,1,2,3), IntArg.create(n)));

        //Print Result
        System.out.println("Result: " + Arrays.toString(out.asIntArray()));
    }
}