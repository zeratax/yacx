public class ExampleSaxpyExecutor {
    public static void main(String[] args){
        int n = 4;
        ArrayArg out = ArrayArg.createOutput(n*4);
        System.out.println(Executor.launch("filter_k", n, 1, out, ValueArg.create(0),
                                                 ArrayArg.create(0,1,2,3), ValueArg.create(n));
        System.out.println("Result: " + out.asFloatArray());
    }
}