//> using scala 2.13
//> using dep "org.pytorch:pytorch_java_only:2.1.0"
//> using javaOpt -Djava.library.path=libtorch/lib

import org.pytorch.Tensor
import org.pytorch.Module

val t = Tensor.fromBlob(Array[Int](1, 2, 3, 4, 5, 6), // data
  Array[Long](2, 3) )

val x = Module.load()




println(t)
println(1+1)
