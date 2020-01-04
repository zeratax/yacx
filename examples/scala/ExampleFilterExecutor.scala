object ExampleFilterExecutor {
    def main(args: Array[String]) : Unit = {
        //Load Libary
        Executor.loadLibary()

        //Create OutputArgument
        val n = 4
        val out = IntArg.createOutput(n/2)
        val counter = IntArg.create(Array[Int](0), true)

        //Compile and launch Kernel
        println("\n" + Executor.launch("filter_k", n, 1, out, counter,
                                                 IntArg.create(0,1,2,3), IntArg.create(n)))

        //Print Result
        println("Result Counter: " + counter.asIntArray()(0))
        println("Result:         [" + out.asIntArray().mkString(", ") + "]")
    }
}