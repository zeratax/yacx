public class ExecutorFailureException extends RuntimeException {
    public ExecutorFailureException(String message){
        super(message);
        System.err.println("Runtime error in the executor");
    }
}