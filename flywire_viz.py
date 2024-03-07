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

DATA_DIR = "../data/Production"
SAVE_DIR = "../output/images"
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

    def get_neuron(self, skeletonize=True, store=False, purge=False, force=False):
        if not force and os.path.exists(
            f"{DATA_DIR}/{self.materialization}_{self.root_id}.zip"
        ):
            print(f"Loading {self.materialization}_{self.root_id} from disk")
            # Pick [0] because it will load as list from .zip
            self.neuron = navis.read_swc(
                f"{DATA_DIR}/{self.materialization}_{self.root_id}.zip"
            )[0]
        else:
            print(f"Downloading {self.root_id} from flywire")
            # self.neuron = flywire.skeletonize_neuron(self.root_id)
            self.neuron = flywire.get_mesh_neuron(self.root_id)
            if skeletonize:
                self.neuron = self.neuron.skeletonize()
            if store:
                self.save_neuron()
        return self.neuron

    def purge_neuron(self):
        print(f"Purging *_{self.root_id} from disk")
        files = glob(f"{DATA_DIR}/*_{self.root_id}.zip")
        for f in files:
            os.remove(f)

    def save_neuron(self, purge=False):
        # TODO: add a purge option that will delete previous materialization of the same root_id
        if purge:
            self.purge_neuron()
        print(f"Saving {self.materialization}_{self.root_id} to disk")
        navis.write_swc(
            self.neuron, f"{DATA_DIR}/{self.materialization}_{self.root_id}.zip"
        )


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

    def get_neurons(self, skeletonize=True, store=False, purge=False, force=False):
        if len(self.ids) == 0:
            self.get_ids()
        for root_id in self.ids:
            neuron = Neuron(root_id, materialization=self.materialization)
            self.neurons[root_id] = neuron.get_neuron(
                store=store, purge=purge, skeletonize=skeletonize, force=force
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
        self.dataset = default_dataset
        self.annotation = annotation_version
        self.materialization = materialization_version

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

        print(f"--- Examining connectivity {self.source_type} > {self.target_type} ---")

    def get_adj(self):
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
        return self.adj

    def plot_targets(self, scene=None):
        if scene is None:
            raise ValueError("Scene must be provided")
        if self.adj is None:
            self.get_adj()

        if not self.adj.empty:
            self.targets.get_neurons(
                store=self.target_store, purge=True, skeletonize=True, force=False
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
                store=self.source_store, purge=True, skeletonize=True, force=False
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

    def set_view3d(self, view=None):
        if view is not None:
            if view == "front":
                view = "XY"
            elif view == "top":
                view = "XZ"
            elif view == "side":
                view = "YZ"
            else:
                raise ValueError(f"Invalid view: {view}")
            self.viewer3d.set_view(view)

    def save3d(self, filename=None):
        if filename is not None:
            data = self.viewer3d.canvas.render()[..., :3]
            plt.imsave(f"{SAVE_DIR}/{filename}", data)

    def plot3d(self):
        self.viewer3d.add(self.brain)
        self.viewer3d.add(self.neurons, color=self.color)
        self.viewer3d.show()

    def close3d(self):
        self.viewer3d.close()


def make_all_LPLC_T(mat="818"):
    if mat == None:
        mat = flywire.get_materialization_versions().loc[0, "id"]
    for vpn in ["LPLC1", "LPLC2"]:
        for src in ["T5", "T4"]:
            for subpre in ["", "a", "b", "c", "d"]:
                # TODO: change to match all directions
                filename = f"{mat}_{src}{subpre}>{vpn}_*.png"
                if len(glob(f"{SAVE_DIR}/{filename}")) == 0:
                    c = Connectivity(
                        f"{src}{subpre}",
                        f"{vpn}",
                        starts_with="none",
                        store="target",
                        source_side="left",
                        target_side="left",
                    )
                    scene = Scene()
                    scene.open3d()
                    c.plot_targets(scene=scene)
                    for dir in ["front", "top", "side"]:
                        filename = f"{mat}_{src}{subpre}>{vpn}_{dir}.png"
                        scene.set_view3d(view=dir)
                        scene.save3d(filename=filename)
                    scene.close3d()


def make_trg_src(
    trg=["LPLC2", "LPLC2", "LPLC1", "LC4", "LPLC4", "LC22", "LC4"],
    src=["Giant Fiber", "PVLP071", "DNp03", "DNp11", "DNp07", "DNp26", "DNp02"],
    mat="818",
):
    if mat == None:
        mat = flywire.get_materialization_versions().loc[0, "id"]
    for sr, tg in zip(src, trg):
        filename = f"{mat}_{sr}>{tg}_*.png"
        if len(glob(f"{SAVE_DIR}/{filename}")) == 0:
            c = Connectivity(
                sr,
                tg,
                starts_with="none",
                store="target",
                source_side="left",
                target_side="left",
            )
            scene = Scene()
            scene.open3d()
            c.plot_targets(scene=scene)
            for dir in ["front", "top", "side"]:
                filename = f"{mat}_{sr}>{tg}_{dir}.png"
                scene.set_view3d(view=dir)
                scene.save3d(filename=filename)
            scene.close3d()


if __name__ == "__main__":
    c = Connectivity(
        ".*(?:Giant Fiber|GF)",
        "^.*LPLC2",
        annotation_version="v2.0.0",
        materialization_version="latest",
        default_dataset="production",
        store="both",
        source_side="right",
        target_side="right",
    )
    scene = Scene()
    scene.open3d()
    c.plot_targets(scene=scene)
    # scene.set_view3d(view=dir)
    input()
    scene.close3d()
