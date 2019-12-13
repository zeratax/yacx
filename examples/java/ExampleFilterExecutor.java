import java.io.IOException;
import java.util.Arrays;

public class ExampleFilterExecutor {
    public static void main(String[] args) throws IOException {
        //Load Libary
        Executor.loadLibary();

        //Create OutputArgument
        int n = 4;
        IntArg out = IntArg.createOutput(n/2);
        IntArg counter = IntArg.createOutput(1);

        //Compile and launch Kernel
        System.out.println("\n" + Executor.launch("filter_k", n, 1, out, counter,
                                                 IntArg.create(0,1,2,3), IntArg.create(n)));

        //Print Result
        System.out.println("Result Counter: " + counter.asIntArray()[0]);
        System.out.println("Result:         " + Arrays.toString(out.asIntArray()));
    }
}