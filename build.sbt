
ThisBuild / scalaVersion     := "2.12.10"

lazy val buildExecutor = taskKey[Unit]("Builds C executor library")

buildExecutor := {
  import scala.language.postfixOps
  import scala.sys.process._
  //noinspection PostfixMethodCall
  "echo y" #| (baseDirectory.value + "/yacx.sh --buildj") !
}

lazy val CUexecutor = (project in file("."))
  .settings(
    name    := "CUDA executor",
    version := "0.5.0",
    libraryDependencies += "junit" % "junit" % "4.11",

    compileOrder := CompileOrder.JavaThenScala,

    compile := ((compile in Compile) dependsOn buildExecutor).value,
    test    := ((test in Test) dependsOn buildExecutor).value
)
