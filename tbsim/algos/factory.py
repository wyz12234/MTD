"""Factory methods for creating models"""
from pytorch_lightning import LightningDataModule
from tbsim.configs.base import ExperimentConfig

from tbsim.algos.algos import (
    BehaviorCloning,
    TransformerTrafficModel,
    TransformerGANTrafficModel,
    VAETrafficModel,
    DiscreteVAETrafficModel,
    BehaviorCloningGC,
    SpatialPlanner,
    GANTrafficModel,
    BehaviorCloningEC,
    TreeVAETrafficModel,
    DiffuserTrafficModel,
    SceneTreeTrafficModel,
    STRIVETrafficModel,
    SceneDiffuserTrafficModel,
)

from models.qc_diffuser_model import QueryDiffuserTrafficModel

from tbsim.algos.multiagent_algos import (
    MATrafficModel,
)

from tbsim.algos.metric_algos import (
    OccupancyMetric
)


def algo_factory(config: ExperimentConfig, modality_shapes: dict):
    """
    A factory for creating training algos

    Args:
        config (ExperimentConfig): an ExperimentConfig object,
        modality_shapes (dict): a dictionary that maps observation modality names to shapes

    Returns:
        algo: pl.LightningModule
    """
    algo_config = config.algo
    algo_name = algo_config.name

    if algo_name == "bc":
        algo = BehaviorCloning(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "bc_gc":
        algo = BehaviorCloningGC(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "vae":
        algo = VAETrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "discrete_vae":
        algo = DiscreteVAETrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "tree_vae":
        if algo_config.scene_centric:
            algo = SceneTreeTrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
        else:
            algo = TreeVAETrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "bc_ec":
        algo = BehaviorCloningEC(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "spatial_planner":
        algo = SpatialPlanner(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "occupancy":
        algo = OccupancyMetric(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "agent_predictor":
        algo = MATrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "TransformerPred":
        algo = TransformerTrafficModel(algo_config=algo_config)
    elif algo_name == "TransformerGAN":
        algo = TransformerGANTrafficModel(algo_config=algo_config)
    elif algo_name == "gan":
        algo = GANTrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "diffuser":
        algo = DiffuserTrafficModel(algo_config=algo_config, modality_shapes=modality_shapes, registered_name=config.registered_name)
    elif algo_name == "strive":
        algo = STRIVETrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "scene_diffuser":
        algo = SceneDiffuserTrafficModel(algo_config=algo_config, modality_shapes=modality_shapes, registered_name=config.registered_name)
    elif algo_name == "query_centric_diffuser":
        algo = QueryDiffuserTrafficModel(algo_config=algo_config, modality_shapes=modality_shapes, registered_name=config.registered_name)
    else:
        raise NotImplementedError("{} is not a valid algorithm" % algo_name)
    return algo
