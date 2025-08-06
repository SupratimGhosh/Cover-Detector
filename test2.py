# Import the InferencePipeline object
import cv2
from inference import InferencePipeline

def my_sink(result, video_frame):
    if result.get("output_image"): # Display an image from the workflow response
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)
    print(result) # do something with the predictions of each frame


# initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="tTodufarW9yFlMaSiAuA",
    workspace_name="trial-kgruu",
    workflow_id="custom-workflow-3",
    video_reference='http://192.168.1.5:8080/video', # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,
    on_prediction=my_sink
)
pipeline.start() #start the pipeline
pipeline.join() #wait for the pipeline thread to finish
