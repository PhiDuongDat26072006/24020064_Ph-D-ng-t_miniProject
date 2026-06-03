import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import traceback
import Checkpoint as Ckp
from Active_modes import Activate_model

if __name__ == '__main__':
    model_path = 'Parameter_cache.pkl'
    # we have types like_____ train, valid, test, model_loss_traking
    type = 'valid'
    model = Ckp.Load_model(model_path)
    try:
        Activate_model(model, type)
    except KeyboardInterrupt:
        if type == "train":
            print("\nĐã dừng training thủ công. Đang lưu khẩn cấp...")
            Ckp.Save_model(model, 'Temp_Parameter.pkl')
        else:
            print("\nĐã dừng quá trình validation/test thủ công")
    except Exception as e:
        print(f"\n[LỖI NGHIÊM TRỌNG]: Chương trình crash vì lỗi sau:")
        traceback.print_exc()