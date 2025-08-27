import os

class GraphSaver:
    """
    A class to save a graph visualization to a file.

    Attributes:
        graph: The graph object to be saved. It should have a method `get_graph()`
               that returns an object with a `draw_mermaid_png()` method to generate
               the PNG image data.
        output_file: The name of the file where the graph will be saved. Defaults to
                     'graph_output.png'.

    Methods:
        save_graph():
            Saves the graph visualization to the specified file. Returns a status
            string indicating whether the file was created, skipped, or if an error
            occurred.
    """
    def __init__(self, graph, output_file="graph_output.png"):
        self.graph = graph
        self.output_file = output_file

    def save_graph(self):
        try:
            # Generate the PNG image data
            graph_png_data = self.graph.get_graph().draw_mermaid_png()

            # Check if the file exists
            if not os.path.exists(self.output_file):
                # Save the image data to a file
                with open(self.output_file, "wb") as f:
                    f.write(graph_png_data)
                print(f"Graph saved as {self.output_file}")
                return "created"
            else:
                print(f"Graph output already exists at {self.output_file}. Skipping creation.")
                return "skipped"
        except Exception as e:
            print(f"An error occurred while saving the graph: {e}")
            return "{e}"

# # Example usage
# if __name__ == "__main__":
#     graph_saver = GraphSaver(graph)
#     result = graph_saver.save_graph()
#     print("Save result:", result)