//> using scala 2.13
//> using dep "org.pytorch:pytorch_java_only:2.1.0"
//> using dep "org.http4s::http4s-dsl:0.23.6"
//> using dep "org.http4s::http4s-blaze-server:0.23.6"
//> using dep "org.http4s::http4s-circe:0.23.6"
//> using dep "io.circe::circe-generic:0.14.1"
//> using dep "org.typelevel::cats-effect:3.3.6"
//> using javaOpt -Djava.library.path=libtorch/lib

import org.pytorch.IValue
import org.pytorch.Module

val module = Module.load("model.pt")

var ans = module.runMethod("forward", IValue.from("dna"))
ans.toDictStringKey.forEach{case (k, v) => println(s"$k: ${v.toDouble}")}
module.runMethod("set_active_items", IValue.listFrom(Seq("dna", "inntrengeren").map(IValue.from):_*))
ans = module.runMethod("forward", IValue.from("dna"), IValue.from(2))

ans.toDictStringKey.forEach{case (k, v) => println(s"$k: ${v.toDouble}")}
