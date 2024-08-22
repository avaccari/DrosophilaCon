import navis
from fafbseg import flywire
import flybrains
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import os
import seaborn as sbn
import numpy as np
from glob import glob
import colorcet as cc
from vispy.util.quaternion import Quaternion

DATA_DIR = "../data/Production"
SAVE_IMG_DIR = "../output/images"
SAVE_CSV_DIR = "../output/csv"
# DATA_DIR = "/Users/avaccari/Library/CloudStorage/GoogleDrive-avaccari@middlebury.edu/Shared drives/MarkD/Flywire/data/Production"
# SAVE_DIR = "/Users/avaccari/Library/CloudStorage/GoogleDrive-avaccari@middlebury.edu/Shared drives/MarkD/Flywire/output/images"
COLORMAP = cc.cm.rainbow_bgyr_35_85_c73
COLORMAP = cc.cm.CET_R2


class Neuron:
    """
    Class to handle neuron data from flywire given a root_id
    """

    def __init__(self, root_id, materialization="latest"):
        self.root_id = root_id
        self.neuron = None
        if materialization == "latest":
            self.materialization = flywire.get_materialization_versions().loc[0, "id"]

    def get_neuron(self, skeletonize=True, store=False, force=False):
        if not force and os.path.exists(
            f"{DATA_DIR}/{self.materialization}_{self.root_id}.zip"
        ):
            print(f"\nLoading {self.materialization}_{self.root_id} from disk")
            # Pick [0] because it will load as list from .zip
            self.neuron = navis.read_swc(
                f"{DATA_DIR}/{self.materialization}_{self.root_id}.zip"
            )[0]
        else:
            print(f"\nDownloading {self.root_id} from flywire")
            # self.neuron = flywire.skeletonize_neuron(self.root_id)
            self.neuron = flywire.get_mesh_neuron(self.root_id)
            if skeletonize:
                print(f" Skeletonizing {self.root_id}")
                self.neuron = self.neuron.skeletonize()
            if store:
                self.save_neuron()
        return self.neuron

    def save_neuron(self, purge=True):
        if purge:
            self.purge_neuron()
        print(f" Saving {self.materialization}_{self.root_id} to disk")
        navis.write_swc(
            self.neuron, f"{DATA_DIR}/{self.materialization}_{self.root_id}.zip"
        )

    def purge_neuron(self):
        print(f"  Purging *_{self.root_id} from disk")
        files = glob(f"{DATA_DIR}/*_{self.root_id}.zip")
        for f in files:
            os.remove(f)
            print(f"Deleted {f}")


class NeuronType:
    """
    Find neurons of a specific type in the flywire database
    """

    def __init__(
        self,
        cell_type,
        side="left",
        limit=None,
        default_dataset="production",
        materialization_version="latest",
        annotation_version="v2.0.0",
    ):
        self.cell_type = cell_type
        self.limit = limit
        self.side = side
        self.ids = []
        self.neurons = {}

        # Setup default values for flywire
        self.dataset = default_dataset
        flywire.set_default_dataset(self.dataset)
        self.annotation = annotation_version
        flywire.set_default_annotation_version(
            self.annotation
        )  # Tag of the latest version
        self.materialization = materialization_version
        self.classification = flywire.get_hierarchical_annotations(
            annotation_version=self.annotation,  # Download the latest version ("Classification" column in Codex)
            materialization=self.materialization,  # Download the latest materialization
            force_reload=False,  # Force to download the data
        )

    def get_ids(self):
        try:
            table_community = flywire.search_community_annotations(
                self.cell_type,
                materialization=self.materialization,
                dataset=self.dataset,
            )
        except Exception:
            unique_ids_community = np.array([], dtype="uint64")
        else:
            unique_ids_community = (
                table_community["root_id"].unique().astype("uint64")
                if len(table_community) > 0
                else np.array([], dtype="uint64")
            )

        try:
            table_hierarchical = flywire.search_annotations(
                self.cell_type,
                annotation_version=self.annotation,
                materialization=self.materialization,
                dataset=self.dataset,
            )
        except Exception:
            unique_ids_hierarchical = np.array([], dtype="uint64")
        else:
            unique_ids_hierarchical = (
                table_hierarchical["root_id"].unique().astype("uint64")
                if len(table_hierarchical) > 0 and table_hierarchical is not None
                else np.array([], dtype="uint64")
            )

        unique_ids_both_sides = np.union1d(
            unique_ids_community, unique_ids_hierarchical
        )

        try:
            classification = flywire.search_annotations(unique_ids_both_sides)
        except Exception:
            classification = None
        else:
            unique_ids = classification.loc[
                classification["side"] == self.side, "root_id"
            ].unique()
            print(
                f"Found {len(unique_ids)} neurons of type {self.cell_type} in side {self.side}"
            )
            if self.limit is not None:
                unique_ids = unique_ids[: self.limit]
            self.ids = unique_ids

        return self.ids

    def get_neurons(self, skeletonize=True, store=False, force=False):
        if len(self.ids) == 0:
            self.get_ids()
        for idx in range(len(self.ids)):
            print(f"Processing neuron {idx + 1} of {len(self.ids)}")
            neuron = Neuron(self.ids[idx], materialization=self.materialization)
            self.neurons[self.ids[idx]] = neuron.get_neuron(
                store=store, skeletonize=skeletonize, force=force
            )
        return self.neurons


class Connectivity:
    """
    Class to handle connectivity between two neuron types
    """

    def __init__(
        self,
        source_type,
        target_type,
        default_dataset="production",
        materialization_version="latest",
        annotation_version="v2.0.0",
        source_side="left",
        target_side="left",
        store="none",
    ):
        # Setup default values for flywire
        self.dataset = default_dataset
        flywire.set_default_dataset(self.dataset)
        self.annotation = annotation_version
        flywire.set_default_annotation_version(
            self.annotation
        )  # Tag of the latest version
        self.materialization = materialization_version
        self.classification = flywire.get_hierarchical_annotations(
            annotation_version=self.annotation,  # Download the latest version ("Classification" column in Codex)
            materialization=self.materialization,  # Download the latest materialization
            force_reload=False,  # Force to download the data
        )

        self.source_type = source_type
        self.target_type = target_type
        self.source_side = source_side
        self.target_side = target_side
        self.sources = None
        self.targets = None
        self.adj = None

        if store == "both":
            self.source_store = True
            self.target_store = True
        elif store == "source":
            self.source_store = True
            self.target_store = False
        elif store == "target":
            self.source_store = False
            self.target_store = True
        elif store == "none":
            self.source_store = False
            self.target_store = False
        else:
            raise ValueError(f"Invalid store value: {store}")

        print(
            f"--- Examining connectivity {self.source_type}_{self.source_side} > {self.target_type}_{self.target_side} ---"
        )

    def get_materialization(self):
        ### Used only as a helper function to get the latest materialization
        ### Not sure if some of the queries can be run with other than
        ### "latest" as materialization
        return flywire.get_materialization_versions().loc[0, "id"]

    def get_adj(self, store=True):
        self.sources = NeuronType(
            self.source_type,
            side=self.source_side,
            default_dataset=self.dataset,
            materialization_version=self.materialization,
            annotation_version=self.annotation,
        )
        self.targets = NeuronType(
            self.target_type,
            side=self.target_side,
            default_dataset=self.dataset,
            materialization_version=self.materialization,
            annotation_version=self.annotation,
        )
        self.adj = flywire.synapses.get_adjacency(
            self.sources.get_ids(),
            targets=self.targets.get_ids(),
            materialization=self.materialization,
            dataset=self.dataset,
            filtered=False,
        )
        if store:
            self.adj.to_csv(
                f"{SAVE_CSV_DIR}/adj_{self.get_materialization()}_{self.source_type}_{self.source_side}>{self.target_type}_{self.target_side}.csv"
            )
        return self.adj

    def plot_targets(self, scene=None):
        if scene is None:
            raise ValueError("Scene must be provided")
        if self.adj is None:
            self.get_adj()

        if not self.adj.empty:
            self.targets.get_neurons(
                store=self.target_store, skeletonize=True, force=False
            )
            targets_sum = self.adj.sum(axis=0).sort_values(ascending=False)
            max_cnt = int(targets_sum.max())
            print(f"Targets' synapses max count: {max_cnt}")
            if max_cnt != 0:
                cmap = mcm.get_cmap(COLORMAP, max_cnt)
                color = {
                    self.targets.neurons[id]: cmap(cnt / max_cnt)
                    for id, cnt in targets_sum.items()
                }
                scene.add_neurons(self.targets, color=color)
                scene.open3d()
                scene.plot3d()

    def plot_sources(self, scene=None):
        if scene is None:
            raise ValueError("Scene must be provided")
        if self.adj is None:
            self.get_adj()
        if not self.adj.empty:
            self.sources.get_neurons(
                store=self.source_store, skeletonize=True, force=False
            )
            sources_sum = self.adj.sum(axis=1).sort_values(ascending=False)
            max_cnt = int(sources_sum.max())
            print(f"Sources' synapses max count: {max_cnt}")
            if max_cnt != 0:
                cmap = mcm.get_cmap(COLORMAP, max_cnt)
                color = {
                    self.sources.neurons[id]: cmap(cnt / max_cnt)
                    for id, cnt in sources_sum.items()
                }
                scene.add_neurons(self.sources, color=color)
                scene.open3d()
                scene.plot3d()

    def plot_source_synapses_hist(self):
        plt.ion()
        if self.adj is None:
            self.get_adj()
        targets_sum = self.adj.sum(axis=1).sort_values(ascending=False)
        targets_sum = targets_sum[targets_sum > 0]
        sbn.histplot(targets_sum)
        plt.title(f"{self.source_type} > {self.target_type}")
        plt.xlabel(f"Number of {self.source_type} synapses")
        plt.ylabel(f"Number of {self.source_type} neurons")
        plt.show()


class Scene:
    def __init__(self, size=(1600, 1200)):
        self.brain = flybrains.FAFB14  #! Might have the wrong units
        self.neurons = []
        self.color = {}
        self.size = size

    def add_brain3d(self):
        state = self.viewer3d.camera3d.get_state()
        self.viewer3d.add(self.brain)
        self.viewer3d.camera3d.set_state(state)

    def open3d(self):
        self.viewer3d = navis.Viewer(size=self.size)

    def add_neurons(self, object, color=None):
        if isinstance(object, NeuronType):
            self.neurons.extend(object.neurons.values())
            self.color = color
        elif isinstance(object, Neuron):
            self.neurons.append(object.neuron)
            self.color[object.neuron] = color
        else:
            raise ValueError(f"Invalid object type: {type(object)}")

    def plot2d(self):
        navis.plot2d([self.brain, self.neurons])
        plt.show()

    def set_view3d(self, view=None, scale_factor=None, center=False):
        if view is None:
            return
        elif isinstance(view, Quaternion):
            pass
        elif view == "front":
            view = "XY"
        elif view == "top":
            view = "XZ"
        elif view == "side":
            view = "YZ"
        else:
            raise ValueError(f"Invalid view: {view}")

        self.viewer3d.set_view(view)

        if center:
            self.viewer3d.center_camera()

        if scale_factor is not None:
            self.viewer3d.camera3d.scale_factor = scale_factor

    def save3d(self, filename=None, dpi=300):
        if filename is not None:
            data = self.viewer3d.canvas.render()[..., :3]
            plt.imsave(f"{SAVE_IMG_DIR}/{filename}", data, dpi=dpi)

    def plot3d(self, add_brain=False):
        self.viewer3d.add(self.neurons, color=self.color)
        if add_brain:
            self.add_brain3d()
        self.viewer3d.show()

    def close3d(self):
        self.viewer3d.close()


def make_src_trg(scrs, trgs, side=["left", "right"], only_adj=False, overwrite=False):
    for sd in side:
        for trg in trgs:
            for src in scrs:
                c = Connectivity(
                    f"{src}",
                    f"{trg}",
                    annotation_version="v2.0.0",
                    materialization_version="latest",
                    default_dataset="production",
                    store="both",
                    source_side=sd,
                    target_side=sd,
                )
                if only_adj:
                    c.get_adj()
                else:
                    mat = c.get_materialization()
                    filename = f"{mat}_{src}_{sd}>{trg}_{sd}.png"
                    if len(glob(f"{SAVE_IMG_DIR}/{filename}")) == 0 or overwrite:
                        scene = Scene()
                        scene.open3d()
                        c.plot_targets(scene=scene)
                        if sd == "left":
                            scene.set_view3d(
                                view=Quaternion(-0.435, -0.151, -0.632, 0.623)
                            )
                        elif sd == "right":
                            # scene.set_view3d(view=Quaternion(0.46, 0.193, -0.649, 0.575))
                            scene.set_view3d(
                                view=Quaternion(-0.643, -0.61, -0.447, 0.12)
                            )
                        scene.add_brain3d()
                        scene.save3d(filename=filename)
                        scene.close3d()


if __name__ == "__main__":
    # make_src_trg(["LPi3-4", "LPi4-3"], ["LPLC2"], ["right"], overwrite=True)
    # make_src_trg(
    #     ["Giant Fiber"], ["LPLC2"], ["right", "left"], only_adj=True, overwrite=True
    # )
    # make_src_trg(
    #     ["LPLC2"], ["Giant Fiber"], ["right", "left"], only_adj=True, overwrite=True
    # )
    # make_src_trg(
    #     ["T4", "T4a", "T4b", "T4c", "T4d", "T5", "T5a", "T5b", "T5c", "T5d"],
    #     ["LPLC2"],
    #     ["right"],
    #     overwrite=True,
    # )
    # make_src_trg(
    #     [
    #         "LPi3-4",
    #         "LPi4-3",
    #         "Y3",
    #         "Tm5f",
    #         "Tm5e",
    #         "Tm36",
    #         "TmY5a",
    #         "Tm7",
    #         "Tm27",
    #         "Tm20",
    #         "Tm16",
    #         "Tm31",
    #         "Tm4",
    #         "Tm3",
    #     ],
    #     ["LPLC2"],
    #     ["left", "right"],
    #     overwrite=False,
    # )
    # make_src_trg(["T4d", "T5d"], ["LLPC3"], ["left", "right"], overwrite=False)
    # make_src_trg(["T4c", "T5c"], ["LLPC2"], ["left", "right"], overwrite=False)
    pass
