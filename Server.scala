//> using scala 2.13
//> using dep "org.pytorch:pytorch_java_only:2.1.0"
//> using dep "org.http4s::http4s-dsl:0.23.6"
//> using dep "org.http4s::http4s-blaze-server:0.23.6"
//> using dep "org.http4s::http4s-circe:0.23.6"
//> using dep "io.circe::circe-generic:0.14.1"
//> using dep "org.typelevel::cats-effect:3.5.2"
//> using javaOpt -Djava.library.path=libtorch/lib

import cats.effect._

import java.time.OffsetDateTime
import scala.collection.JavaConverters._
import scala.concurrent.duration.{FiniteDuration, _}
import org.http4s._
import org.http4s.blaze.server.BlazeServerBuilder
import org.http4s.circe.CirceEntityCodec.circeEntityEncoder
import org.http4s.dsl.io._
import org.http4s.implicits._
import org.pytorch.{IValue, Module}


object ContentIdQueryParamMatcher extends QueryParamDecoderMatcher[String]("contentId")
object Async2 extends IOApp {
  class ModelRepo(cache: Ref[IO, Module]) {
    def asset(name: String): IO[Module] = cache.get
  }

  object ModelRepo extends {
    def load(current: Option[OffsetDateTime]): IO[Module] =
      IO.blocking {
        println("Called Load")
        Module.load("model.pt")
      }

    def resource: Resource[IO, ModelRepo] =
      for {
        ref <- Resource.eval(Ref.of[IO, Module](Module.load("model.pt")))
        sig <- Resource.eval(fs2.concurrent.SignallingRef.of[IO, Boolean](false))
        runUpdate =
          (fs2.Stream.emit[IO, FiniteDuration](0.second) ++ fs2.Stream.awakeEvery[IO](10.seconds))
            .interruptWhen(sig)
            .evalMap(prevModel => load(Some(OffsetDateTime.now())))
            .evalTap(newModel => ref.set(newModel))
            .compile
            .drain
            .start
        cache <- Resource.make(runUpdate.as(new ModelRepo(ref)))(_ => sig.set(true))
      } yield cache

  }

  def modelService(mycache: ModelRepo) = HttpRoutes.of[IO] {
    case GET -> Root => {
      Ok(s"Hello, World!")
    }
    case GET -> Root / recommenderType / "mlt" :? ContentIdQueryParamMatcher(contentId) =>
      val out = for {
        module <- mycache.asset("a")
        recs = module.runMethod("forward", IValue.from(contentId))
        resp = recs.toDictStringKey.asScala.map { case (k, v) => (k, v.toDouble) }.toMap

      } yield (resp)
      Ok(out)
  }

  override def run(args: List[String]): IO[ExitCode] = {
    println("try to get cache")
    ModelRepo.resource.use{ cache =>
      BlazeServerBuilder[IO]
        .bindHttp(8080, "localhost")
        .withHttpApp(modelService(cache).orNotFound)
        .serve
        .compile
        .drain
        .as(ExitCode.Success)
    }
  }
}