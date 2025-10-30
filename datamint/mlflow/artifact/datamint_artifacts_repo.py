from mlflow.store.artifact.mlflow_artifacts_repo import MlflowArtifactsRepository


class DatamintArtifactsRepository(MlflowArtifactsRepository):
    @classmethod
    def resolve_uri(cls, artifact_uri, tracking_uri):
        tracking_uri = tracking_uri.split('datamint://', maxsplit=1)[-1]
        return super().resolve_uri(artifact_uri, tracking_uri)
