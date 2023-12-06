//> using scala 2.13
//> using dep "org.pytorch:pytorch_java_only:2.1.0"
//> using javaOpt -Djava.library.path=libtorch/lib

/*
std::unordered_map<std::string, torch::jit::IValue> umap = {{"x", myIvalue}, {"opt", myIvalue2}};
auto result = module.get_method("forward")({}, umap);
// shows all potential arguments to model forward
std::cout << module.get_method("forward").function().getSchema().arguments() << std::endl;
 */

import org.pytorch.Tensor
import org.pytorch.IValue
import org.pytorch.Module

val t = Tensor.fromBlob(Array[Int](1, 2, 3, 4, 5, 6), // data
  Array[Long](2, 3) )

val module = Module.load("model.pt")
val ans = module.runMethod("forward", IValue.from("dna"))

ans.toDictStringKey.forEach{case (k, v) => println(s"$k: ${v.toDouble}")}


println(t)
println(1+1)
