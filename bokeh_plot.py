import numpy as np
from bokeh.plotting import ColumnDataSource, figure, output_file, show, output_notebook
import umap
import numpy as np
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import output_notebook

# Load the embeddings from a file (replace 'embeddings.txt' with your file path)
embeddings_file_path = 'combined_0.3_hin.tsv'  # Replace with your file path

# Read the embeddings data
embeddings = []
with open(embeddings_file_path, 'r') as file:
    for line in file:
        embedding = list(map(float, line.strip().split('\t')))
        embeddings.append(embedding)

# Convert to NumPy array
embeddings = np.array(embeddings)

# Example: List of image paths (for 10 embeddings) and textual descriptions (for 20 embeddings)
img_root_path = './images/'
image_names = ['jalebi.jpg',\
               'Gujiya.jpg', \
               'hasua.jpg', \
               'bullock_cart.jpg',\
               'padmanabha_swamy_temple.jpg',\
               'Ghatam.jpg', \
               'Kathakali.jpg',\
               'kumbh.jpg',\
               'durga_puja.jpg',\
               'Barsana_Holi_Festival.jpg']
image_paths = [img_root_path + i for i in image_names]


# Replace 'your_file.txt' with the path to your file
file_path = './text_desc.txt'

# Read the lines of the file into a list
with open(file_path, 'r') as file:
    lines = file.readlines()

# Strip newline characters from each line (optional)
lines = [line.strip() for line in lines]
lines = lines[:10]
# Print the list of lines
print(lines)

text_descriptions =  lines + lines # Replace with your actual text descriptions

# Concatenate image paths and text descriptions with labels
# Assuming images are the first 10 embeddings and texts are the next 20
labels = '1_img','2_img','3_img','4_img','5_img','6_img','7_img','8_img','9_img','10_img','1_non_aligned_text','2_non_aligned_text','3_non_aligned_text','4_non_aligned_text','5_non_aligned_text','6_non_aligned_text','7_non_aligned_text','8_non_aligned_text','9_non_aligned_text','10_non_aligned_text','1_text','2_text','3_text','4_text','5_text','6_text','7_text','8_text','9_text','10_text'
hover_info = image_paths + text_descriptions  # List containing either image paths or text descriptions

# Verify that you have 30 embeddings, 10 image paths, and 20 text descriptions
assert len(embeddings) == len(labels) == len(hover_info), "Mismatch in number of embeddings, labels, or hover info."

# Apply UMAP for dimensionality reduction to 2D
reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)



# Activate Bokeh to render in notebook if using Jupyter
output_notebook()


# Create a ColumnDataSource with the reduced embeddings, labels, and hover info
source = ColumnDataSource(data=dict(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    label=labels,
    hover_info=hover_info,
))

# Create a Bokeh plot
plot = figure(title="UMAP Projection with Image and Text Hover",
              width=800, height=600,
              tools="pan,wheel_zoom,reset")


plot.circle('x', 'y', size=10, source=source, color="blue", alpha=0.6, legend_label='Images')
plot.triangle('x', 'y', size=10, source=source, color="green", alpha=0.6, legend_label='Texts')


# # Create a hover tool that will display the hover information for each point
hover = HoverTool(
    tooltips="""
    <div>
    <div>@label</div>
        <div>
            <img src="@hover_info" alt="Image" width="100" height="100" style="border: 2px solid black;" />
        </div>
        <div>@hover_info</div>
    </div>
"""
)

# hover = HoverTool(
#     tooltips=[
#         ("Label", "@label"),
#         ("Hover Info", "@hover_info")
#     ]
# )

# Add the hover tool to the plot
plot.add_tools(hover)

# Show the output in a standalone HTML file or inline in Jupyter
output_file("umap_with_image_text_hover.html")
show(plot)
