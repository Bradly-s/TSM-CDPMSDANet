import visualdl
from visualdl.server import app


logdir = "/mnt/sdb1/swf/project/PaddleVideo/log/log_modify_my.txt"
visualdl.server.app.run(logdir,
                        model="/mnt/sdb1/swf/project/PaddleVideo/output/ppTSM_task_VideoClassfy_frames_dense_no_pretrained_modify/ppTSM_task_VideoClassfy_frames_dense_no_pretrained_modify_best.pdopt",
                        host="127.0.0.1",
                        port=8080,
                        cache_timeout=20,
                        language=None,
                        public_path=None,
                        api_only=False,
                        open_browser=False)