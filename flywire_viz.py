import navis
from fafbseg import flywire
import flybrains
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import seaborn as sbn

DATA_DIR = "../data/FlyWire/Production"
SAVE_DIR = "../output/images"
# DATA_DIR = "/Users/avaccari/Library/CloudStorage/GoogleDrive-avaccari@middlebury.edu/Shared drives/MarkD/Analysis/data/FlyWire/Production"


class Neuron:
    def __init__(self, neuron_id):
        self.neuron_id = neuron_id
        self.neuron = None

    def get_neuron(self, store=False):
        if os.path.exists(f"{DATA_DIR}/{self.neuron_id}.zip"):
            print(f"Loading {self.neuron_id} from disk")
            # Pick [0] because it will load as list from .zip
            self.neuron = navis.read_swc(f"{DATA_DIR}/{self.neuron_id}.zip")[0]
        else:
            print(f"Downloading {self.neuron_id} from flywire")
            self.neuron = flywire.skeletonize_neuron(
                self.neuron_id, dataset="production"
            )
            if store:
                self.save_neuron()
        return self.neuron

    def save_neuron(self):
        print(f"Saving {self.neuron.id} to disk")
        navis.write_swc(self.neuron, f"{DATA_DIR}/{self.neuron.id}.zip")


class NeuronType:
    def __init__(self, cell_type, starts_with=True, limit=None):
        self.cell_type = f"^{cell_type}" if starts_with else cell_type
        self.starts_with = starts_with
        self.limit = limit
        self.ids = []
        self.neurons = {}

    def get_ids(self):
        table = flywire.find_celltypes(self.cell_type)
        unique_ids = table["root_id"].sort_values().unique()
        print(f"Found {len(unique_ids)} neurons of type {self.cell_type}")
        if self.limit is not None:
            unique_ids = unique_ids[: self.limit]
        self.ids = unique_ids
        return self.ids

    def get_neurons(self, store=False):
        if len(self.ids) == 0:
            self.get_ids()
        for neuron_id in self.ids:
            neuron = Neuron(neuron_id)
            self.neurons[neuron_id] = neuron.get_neuron(store=store)
        return self.neurons


class Connectivity:
    def __init__(self, source_type, target_type, starts_with="both", store="none"):
        self.source_type = source_type
        self.target_type = target_type
        self.sources = None
        self.targets = None
        self.adj = None

        if starts_with == "both":
            self.source_starts_with = True
            self.target_starts_with = True
        elif starts_with == "source":
            self.source_starts_with = True
            self.target_starts_with = False
        elif starts_with == "target":
            self.source_starts_with = False
            self.target_starts_with = True
        elif starts_with == "none":
            self.source_starts_with = False
            self.target_starts_with = False
        else:
            raise ValueError(f"Invalid starts_with value: {starts_with}")

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

    def get_adj(self):
        self.sources = NeuronType(self.source_type, starts_with=self.source_starts_with)
        self.targets = NeuronType(self.target_type, starts_with=self.target_starts_with)
        self.adj = flywire.synapses.fetch_adjacency(
            self.sources.get_ids(), targets=self.targets.get_ids()
        )
        return self.adj

    def plot_targets(self, view=None, filename=None, scene=None):
        if scene is None:
            scene = Scene()
        if self.adj is None:
            self.get_adj()
        self.targets.get_neurons(store=self.target_store)
        targets_sum = self.adj.sum(axis=0).sort_values(ascending=False)
        max_cnt = int(targets_sum.max())
        viridis = cm.get_cmap("viridis", max_cnt)
        color = {
            self.targets.neurons[id]: viridis(cnt / max_cnt)
            for id, cnt in targets_sum.items()
        }
        scene.add_neurons(self.targets, color=color)
        scene.open3d()
        scene.plot3d(view=view, filename=filename)

    def plot_sources(self, view=None, filename=None, scene=None):
        if scene is None:
            scene = Scene()
        if self.adj is None:
            self.get_adj()
        self.sources.get_neurons(store=self.source_store)
        sources_sum = self.adj.sum(axis=1).sort_values(ascending=False)
        max_cnt = int(sources_sum.max())
        viridis = cm.get_cmap("viridis", max_cnt)
        color = {
            self.sources.neurons[id]: viridis(cnt / max_cnt)
            for id, cnt in sources_sum.items()
        }
        scene.add_neurons(self.sources, color=color)
        scene.open3d()
        scene.plot3d(view=view, filename=filename)

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

    # TODO add plot_sources and try LPLC2>Giant Fiber or LC4 dnp11 or dnp02
    # assuming they exist.


class Scene:
    def __init__(self):
        self.brain = flybrains.FAFB14  #! Might have the wrong units
        self.neurons = []
        self.color = {}

    def open3d(self):
        self.viewer3d = navis.Viewer()

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

    def plot3d(self, view=None, filename=None):
        self.viewer3d.add(self.brain)
        self.viewer3d.add(self.neurons, color=self.color)
        self.viewer3d.show()
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
        if filename is not None:
            data = self.viewer3d.canvas.render()[..., :3]
            plt.imsave(f"{SAVE_DIR}/{filename}", data)

    def close3d(self):
        self.viewer3d.close()


if __name__ == "__main__":
    for vpn in ["LPLC1", "LPLC2"]:
        for side in ["R", "L"]:
            for pre in ["T5", "T4"]:
                for subpre in ["", "a", "b", "c", "d"]:
                    for dir in ["front", "top", "side"]:
                        filename = f"{vpn}_{side}_{pre}{subpre}_{dir}.png"
                        if not os.path.exists(f"{SAVE_DIR}/{filename}"):
                            scene = Scene()
                            scene.open3d()
                            c = Connectivity(
                                f"{pre}{subpre}",
                                f"{vpn}_{side}",
                                starts_with="both",
                                store="target",
                            )
                            c.plot_targets(view=dir, filename=filename, scene=scene)
                            scene.close3d()
