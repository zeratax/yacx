public class ExecutorFailureException extends Exception {
    public ExecutorFailureException(String message){
        super(message);
        System.err.println("Runtime error in the executor");
    }
}