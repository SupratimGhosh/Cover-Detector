import cv2
import threading
import os
from dotenv import load_dotenv
from inference import InferencePipeline

load_dotenv()

# Global flag to signal threads to exit
exit_flag = threading.Event()

def my_sink(result, video_frame):
    if result.get("output_image"):
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag.set()
            cv2.destroyAllWindows()

    # Print only class and confidence for each prediction
    predictions = result.get("model_predictions", {}).get("predictions", [])
    for pred in predictions:
        print(f"class: {pred.get('class')}, confidence: {pred.get('confidence'):.4f}")

    

def video_feed_thread(video_url):
    cap = cv2.VideoCapture(video_url)
    while cap.isOpened() and not exit_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Raw Video Feed", frame)
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag.set()
            cv2.destroyAllWindows()
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    video_url = 0
    # Start video feed in parallel thread
    t = threading.Thread(target=video_feed_thread, args=(video_url,), daemon=True)
    t.start()

    # Start inference pipeline
    pipeline = InferencePipeline.init_with_workflow(
        api_key=os.getenv("ROBOFLOW_API_KEY"),
        workspace_name="trial-kgruu",
        workflow_id="custom-workflow-2",
        video_reference=video_url, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
        max_fps=30,
        on_prediction=my_sink
    )
    pipeline.start()
    # Wait for exit_flag to be set
    while not exit_flag.is_set():
        pass
    pipeline.join()

if __name__ == "__main__":
    main()
