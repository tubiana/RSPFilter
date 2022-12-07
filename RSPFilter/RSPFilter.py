import starfile
import os, sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import argparse
from scipy.stats import binned_statistic_2d
import panel as pn
import warnings
warnings.filterwarnings("ignore")
np.seterr(divide = 'ignore') 



try:
    # This if the "builded" import version
    from .version import __version__
except:
    # for usage from sources
    from version import __version__


pn.extension(loading_spinner="dots")


def parse_arg():
    parser = argparse.ArgumentParser(
        description="Homogeneous particle filtering of STAR file"
    )

    parser.add_argument("-f", "--file", help="StarFile")
    parser.add_argument(
        "-r", "--resolution", help="Grid density (default:5)", default=5
    )
    parser.add_argument(
        "-d", "--threshold", help="density threshold (default:100)", default=100
    )
    parser.add_argument("-o", "--output", help="outputfile", default=None)
    parser.add_argument(
        "-g", "--gui", help="start a webserver interactive GUI (Y/n)", default="Y"
    )
    parser.add_argument("-c1","--column1", help="Column 1 in the starfile", default="rlnAngleRot")
    parser.add_argument("-c2","--column2", help="Column 2 in the starfile", default="rlnAngleTilt")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = vars(parser.parse_args())
    return args


def compute_density_matrix(df, resolution, col1, col2):
    """
    Compute the density matrix of particles depending a grid resolution (in angle)
    Args:
        - df <pd.DataFrame>: Pandas dataframe of all particles (output for starfile["particles"])
        - resolution <int>: resolution of the grid (in angle)
    Returns:
        - df <pd.DataFrame>: Pandas dataframe of all particles (output for starfile["particles"]) with the grid location (gridX, gridY) and density value (density)
        - density <np.array>: density grid matrix.
    """

    global xbins  # I know, it's cheating....
    global ybins
    print(col1,col2)
    xmin, xmax, ymin, ymax = (
        df[col1].min(),
        df[col1].max(),
        df[col2].min(),
        df[col2].max(),
    )
    meshgrid_ResolutionX = int(np.floor((abs(xmin) + abs(xmax)) / resolution))
    meshgrid_ResolutionY = int(np.floor((abs(ymin) + abs(ymax)) / resolution))

    xbins = np.linspace(xmin, xmax, meshgrid_ResolutionX + 1)
    ybins = np.linspace(ymin, ymax, meshgrid_ResolutionY + 1)

    ret = binned_statistic_2d(
        df[col1],
        df[col2],
        None,
        "count",
        bins=[xbins, ybins],
        expand_binnumbers=True,
    )
    gridX = ret.binnumber[0] - 1
    gridY = ret.binnumber[1] - 1
    density = ret.statistic

    df["gridX"] = gridX
    df["gridY"] = gridY
    df["density"] = density[gridX, gridY]
    density = density.T

    return df, density


def cutoff_values(df, density_threshold, col1, col2):
    """
    Function to remove randomly particle where their square in the grid has a density above a density threshold.
    Args:
        - df <pd.DataFrame>: Pandas dataframe of all particles (output for starfile["particles"])
        - density <np.array>: Density grid. Matrix of size N,M (depending the grid resolution) and each value represent the number of particle in the grid square
        - density_threshold <int>: Density threshold.
    Returns:
        - new_density <np.array>: New density grid (once the particles are filtered)
        - results <pd.DataFrame>: The filtered particles dataframe.
    """

    def remove_random(group, density_threshold):
        density = group.density.unique()[0]
        num_to_remove = int(density - density_threshold)
        indices_to_remove = np.random.choice(group.index, num_to_remove, replace=False)
        group = group.drop(indices_to_remove)
        return group

    results = (
        df.query("density > @density_threshold")
        .groupby(["gridX", "gridY"], as_index=False)
        .apply(lambda x: remove_random(x, density_threshold))
    )

    results = pd.concat([df.query("density <=@density_threshold"), results])
    results = results.reset_index(drop=True)
    print(f"number of particles after cutoff : {len(results)}")

    new_density = binned_statistic_2d(
        results[col1],
        results[col2],
        None,
        "count",
        bins=[xbins, ybins],
        expand_binnumbers=True,
    ).statistic
    new_density = new_density.T

    return new_density, results


def save_starfile(star, results, outputname="output.star"):
    """
    Save the output star file without the added columns
    Args:
        - star <starfile>: input starfile as a "starfile" object
        - results <pd.DataFrame>: result filtered dataframe
        - outputname <string>: output star filename
    """
    results = results.drop(["gridX", "gridY", "density"], axis=1)
    star["particles"] = results

    starfile.write(star, outputname, overwrite=True)


def do_graph(data, aslog=False, surfaceplot=False, title="Density grid"):
    if aslog:
        dataplot = np.where(data > 0, np.log(data), 0)
    else:
        dataplot = data

    if surfaceplot == False:
        fig = px.imshow(data)
    else:
        fig = go.Figure(data=[go.Surface(z=data)])

    fig.update_layout(width=600, height=600, margin=dict(l=50, r=10, b=10, t=50))
    fig.update_layout(title=title)
    return fig


class UI:
    def __init__(
        self,
        star,
        resolution,
        density_threshold,
        outputFile,
        df,
        density,
        new_density,
        results,
        col1, 
        col2,
    ):
        self.star = star
        self.resolution = resolution
        self.density_threshold = density_threshold
        self.outputFile = outputFile
        self.df = df
        self.density = density
        self.new_density = new_density
        self.results = results
        self.col1 = col1,
        self.col2 = col2,
        self.columnList = list(df.select_dtypes(include = ['float']).columns)

        self.fig1 = do_graph(
            self.density, aslog=False, surfaceplot=False, title="density before cleanup"
        )
        self.fig2 = do_graph(
            self.new_density,
            aslog=False,
            surfaceplot=False,
            title="density After cleanup",
        )
        self.fig3D1 = do_graph(
            self.density, aslog=False, surfaceplot=True, title="density before cleanup"
        )
        self.fig3D2 = do_graph(
            self.new_density,
            aslog=False,
            surfaceplot=True,
            title="density After cleanup",
        )

        self.instance_ui()

    def start(self):
        self.mainUI.show()

    def instance_ui(self):
        self.Debug = pn.widgets.TextEditor(name="debug")

        self.scale_log_fig1 = pn.widgets.Checkbox(name="Show axis as log?", value=False)
        self.scale_log_fig2 = pn.widgets.Checkbox(name="Show axis as log?", value=False)
        self.resSlider = pn.widgets.IntSlider(
            start=1,
            end=50,
            step=1,
            value=self.resolution,
            name="Resolution",
        )
        self.densitySlider = pn.widgets.IntSlider(
            start=1,
            end=1000,
            step=1,
            value=self.density_threshold,
            name="Density threshold",
        )

        self.outputPathWidgets = pn.widgets.TextInput(
            name="Output file", value=self.outputFile
        )

        self.updateButton = pn.widgets.Button(name="Update", button_type="primary")
        self.saveButton = pn.widgets.Button(name="Save", button_type="success")
        self.totalN = pn.widgets.StaticText(
            name="Total number of particles", value=f"{len(self.df)}"
        )
        self.cleanN = pn.widgets.StaticText(
            name="Reduced number of particles", value=f"{len(self.results)}"
        )
        # self.cleanNDial = pn.indicators.Dial(name="N Particles",value=len(self.results), bounds=(0,len(self.df)), format='{value}')
        # self.NumberParticleIndicator = pn.indicators.LinearGauge(name="N Particles",value=len(self.results), bounds=(0,len(self.df)), format='{value}', horizontal=True, height=800)
        self.NumberParticleIndicator = pn.indicators.Dial(
            name="N Particles",
            value=len(self.results),
            bounds=(0, len(self.df)),
            format="{value}",
            align="center",
        )

        self.graph1 = pn.pane.Plotly(self.fig1)
        self.graph3D1 = pn.pane.Plotly(self.fig3D1)
        self.graph2 = pn.pane.Plotly(self.fig2)
        self.graph3D2 = pn.pane.Plotly(self.fig3D2)

        #Column dropdown
        self.column1Selector = pn.widgets.Select(name="Column1",options=self.columnList, value='rlnAngleRot')
        self.column2Selector = pn.widgets.Select(name="Column2",options=self.columnList, value='rlnAngleTilt')
        self.test = pn.widgets.Select(name="test",options=["A","B","C","D"], value="B")
        self.mainUI = pn.layout.Column(
            pn.Card(
                pn.layout.Row(
                    pn.layout.Column(
                        self.resSlider,
                        self.densitySlider,
                        pn.Row(
                            self.column1Selector,
                            pn.widgets.StaticText(name="vs"),
                            self.column2Selector,
                        ),
                        self.updateButton,
                        self.outputPathWidgets,
                        self.saveButton,
                     
                        sizing_mode="stretch_width",

                    ),
                    self.NumberParticleIndicator,
                    sizing_mode="stretch_both",
                ),
                title="Parameters",
                collapsed=False,
                collapsible=False,
                align="center",
                
            ),
            pn.Card(
                pn.layout.Divider(),
                pn.layout.Row(
                    pn.layout.Column(
                        self.scale_log_fig1,
                        pn.Tabs(("2D Graph", self.graph1), ("3D Graph", self.graph3D1)),
                    ),
                    pn.layout.Column(
                        self.scale_log_fig2,
                        pn.Tabs(("2D Graph", self.graph2), ("3D Graph", self.graph3D2)),
                    ),
                ),

                title="Graphics",
                collapsed=False,
                collapsible=False,
            ),
            width=800,
            
        )
        self.column1Selector.param.watch(self.update, "value")
        self.column2Selector.param.watch(self.update, "value")

        self.scale_log_fig1.param.watch(self.update_fig, "value")
        self.scale_log_fig2.param.watch(self.update_fig, "value")

        self.updateButton.on_click(self.update)
        self.saveButton.on_click(self.save)

    def update(self, event):
        self.mainUI.loading = True

        self.col1 = self.column1Selector.value
        self.col2 = self.column2Selector.value

        self.df, self.density = compute_density_matrix(self.df, self.resSlider.value, self.col1, self.col2)
        self.new_density, self.results = cutoff_values(
            self.df, self.densitySlider.value,  self.col1, self.col2,
        )

        self.cleanN.value = len(self.results)
        self.NumberParticleIndicator.value = len(self.results)

        self.update_fig(None)
        self.mainUI.loading = False

    def save(self, event):
        self.mainUI.loading = True
        save_starfile(self.star, self.results, self.outputPathWidgets.value)
        self.mainUI.loading = False

    def update_fig(self, event):

        if self.scale_log_fig1.value == True:
            self.newdata1 = np.where(self.density > 0, np.log(self.density), 0)
            self.fig1.data[0]["z"] = self.newdata1
            self.fig3D1.data[0]["z"] = self.newdata1
        else:
            self.fig1.data[0]["z"] = self.density
            self.fig3D1.data[0]["z"] = self.density

        if self.scale_log_fig2.value == True:
            self.newdata2 = np.where(self.new_density > 0, np.log(self.new_density), 0)
            self.fig2.data[0]["z"] = self.newdata2
            self.fig3D2.data[0]["z"] = self.newdata2
        else:
            self.fig2.data[0]["z"] = self.new_density
            self.fig3D2.data[0]["z"] = self.new_density

        self.graph1.object = self.fig1
        self.graph3D1.object = self.fig3D1

        self.graph2.object = self.fig2
        self.graph3D2.object = self.fig3D2


def main():
    print(
        f"""
'########:::'######::'########::'########:'####:'##:::::::'########:'########:'########::
 ##.... ##:'##... ##: ##.... ##: ##.....::. ##:: ##:::::::... ##..:: ##.....:: ##.... ##:
 ##:::: ##: ##:::..:: ##:::: ##: ##:::::::: ##:: ##:::::::::: ##:::: ##::::::: ##:::: ##:
 ########::. ######:: ########:: ######:::: ##:: ##:::::::::: ##:::: ######::: ########::
 ##.. ##::::..... ##: ##.....::: ##...::::: ##:: ##:::::::::: ##:::: ##...:::: ##.. ##:::
 ##::. ##::'##::: ##: ##:::::::: ##:::::::: ##:: ##:::::::::: ##:::: ##::::::: ##::. ##::
 ##:::. ##:. ######:: ##:::::::: ##:::::::'####: ########:::: ##:::: ########: ##:::. ##:
..:::::..:::......:::..:::::::::..::::::::....::........:::::..:::::........::..:::::..::

RSPFilter v {__version__} - Relion Star file Particles Filter 
"""
    )
    args = parse_arg()

    inputfile = args["file"]
    resolution = args["resolution"]
    density_threshold = args["threshold"]
    outputFile = args["output"]
    useGUI = args["gui"]
    col1 = args["column1"]
    col2 = args["column2"]
    useGUI = True if useGUI == "Y" else False

    if not os.path.isfile(inputfile):
        print("No file found. Please check your file again")
        sys.exit(1)

    from pathlib import Path

    if outputFile == None:
        fullPath = Path(inputfile).resolve()
        folder = str(fullPath.parents[0])
        name = str(fullPath.stem)
        outputFile = folder + "/" + name + "_filtered.star"

    # Read star file
    print("star reading... Please wait")
    star = starfile.read(inputfile)
    print("> Done")

    # Get the particle dataframe from the star file
    df = star["particles"]

    # First compute the corresponding density grid and filtered dataset from current parameters
    df, density = compute_density_matrix(df, resolution, col1, col2)
    new_density, results = cutoff_values(df, density_threshold, col1, col2)

    # Create GUI if we want a GUI, otherwise save the new star file directly.
    if useGUI:
        webApp = UI(
            star=star,
            resolution=resolution,
            density_threshold=density_threshold,
            outputFile=outputFile,
            df=df,
            density=density,
            new_density=new_density,
            results=results,
            col1=col1,
            col2=col2,
        )
        webApp.start()
    else:
        print(f"NO WebAPP mode. Saving the new starfile as {outputFile}")
        save_starfile(star, results, outputFile)
        print("> Done.")


if __name__ == "__main__":
    main()
