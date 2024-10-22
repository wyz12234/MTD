"""DataModule / Dataset factory"""
from tbsim.utils.config_utils import translate_trajdata_cfg, translate_pass_trajdata_cfg
from tbsim.datasets.trajdata_datamodules import UnifiedDataModule, PassUnifiedDataModule

from utils.config_utils import translate_query_centric_trajdata_cfg
from datasets.query_centric_datamoule import QCDiffuserDataModule

def datamodule_factory(cls_name: str, config):
    """
    A factory for creating pl.DataModule.

    Valid module class names: "L5MixedDataModule", "L5RasterizedDataModule"
    Args:
        cls_name (str): name of the datamodule class
        config (Config): an Experiment config object
        **kwargs: any other kwargs needed by the datamodule

    Returns:
        A DataModule
    """
    if cls_name.startswith("Unified"):
        trajdata_config = translate_trajdata_cfg(config)
        datamodule = eval(cls_name)(data_config=trajdata_config, train_config=config.train)
    elif cls_name.startswith("PassUnified"):
        trajdata_config = translate_pass_trajdata_cfg(config)
        datamodule = eval(cls_name)(data_config=trajdata_config, train_config=config.train)
    elif cls_name == "QCDiffuserDataModule":
        trajdata_config = translate_query_centric_trajdata_cfg(config)
        datamodule = QCDiffuserDataModule(data_config=trajdata_config, train_config=config.train)
    else:
        raise NotImplementedError("{} is not a supported datamodule type".format(cls_name))
    return datamodule