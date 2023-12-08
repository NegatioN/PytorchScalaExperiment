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

object Server extends IOApp {

  object ContentIdQPM extends QueryParamDecoderMatcher[String]("contentId")
  object UserIdQPM extends QueryParamDecoderMatcher[String]("userId")
  object SizeQPM extends OptionalQueryParamDecoderMatcher[Int]("size")
  object CandidatesQPM extends OptionalQueryParamDecoderMatcher[String]("candidates")


  case class RecommendRequest(id: String, size: Int, candidates: Option[Set[String]])

  class ModelRepo(cache: Ref[IO, Recommender]) {
    // TODO have several recommenders in a map
    def asset(name: String): IO[Recommender] = cache.get
  }

  object ModelRepo extends {
    def load(current: Option[OffsetDateTime]): IO[Recommender] =
      IO.blocking {
        // TODO download a model
        Recommender(Module.load("model.pt"), Set.empty)
      }

    def resource: Resource[IO, ModelRepo] =
      for {
        ref <- Resource.eval(Ref.of[IO, Recommender](Recommender(Module.load("model.pt"), Set.empty)))
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

  val defaultSize = 5
  case class Recommender(model: Module, strategy: Set[String])

  def recommend(rr: RecommendRequest, rec: Recommender): Seq[(String, Double)] = {
    val recs = rec.model.runMethod("forward", IValue.from(rr.id), IValue.from(rr.size))
    recs.toDictStringKey.asScala.map { case (k, v) => (k, v.toDouble) }.toSeq.sortBy(-_._2)
  }

  def modelService(modelRepo: ModelRepo) = HttpRoutes.of[IO] {
    case GET -> Root => Ok(s"Hello, World!")

    case GET -> Root / "mlt" / recommenderName :? ContentIdQPM(contentId) +& SizeQPM(size) +& CandidatesQPM(candidates) =>
      val rr = RecommendRequest(contentId, size.getOrElse(defaultSize), candidates.map(_.split(",").toSet))
      val resp = for {
        recommender <- modelRepo.asset(recommenderName)
      } yield recommend(rr, recommender)
      Ok(resp)

    case GET -> Root / "recommend" / recommenderName :? UserIdQPM(userId) +& SizeQPM(size) +& CandidatesQPM(candidates) =>
      val rr = RecommendRequest(userId, size.getOrElse(defaultSize), candidates.map(_.split(",").toSet))
      val resp = for {
        recommender <- modelRepo.asset(recommenderName)
      } yield recommend(rr, recommender)
      Ok(resp)

    case GET -> Root / "_" / "health" => Ok("Healthy") // TODO ensure we have recommenders loaded.
  }

  override def run(args: List[String]): IO[ExitCode] = {
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