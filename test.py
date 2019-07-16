import torch
import numpy as np
from torch.autograd import Variable


def test(args,
         test_dataloader,
         model, epoch, i):
    try:
        model.eval()
        # ---------------------------     Steps computaion    ----------------------------------
        total_tested_samples = 0  # to calculate the average of accuracy
        total_right_pred = 0
        static_matrix = np.zeros([args.num_class,args.num_class])

        with torch.no_grad():
            for index, (batch_data, batch_label) in enumerate(test_dataloader):
                inputs = Variable(batch_data.to(args.device))
                labels = Variable(batch_label.to(args.device))

                total_tested_samples += labels.size(0)

                outputs = model(inputs)
                _, prediction = torch.max(outputs.data,1)

                total_right_pred += (prediction == labels).sum().item()
                # ------------     True and Prediction matrix value static    ---------------------
                """
                4x4 matrix
                    row replaces the true labels
                    column represents the predication class
                    tp,tn
                    fp.fn
                """
                for num in range(len(labels)):
                    static_matrix[labels[i].cpu().numpy(),
                                  prediction[i].cpu().numpy()] += 1.0
            test_acc = total_right_pred/total_tested_samples
            print('Test:[{}/{}]-Epoch:[{}/{}]--Acc[{:.4f}]'.
                  format(i, args.k, epoch, args.epochs, test_acc))
        return test_acc, static_matrix
    except Exception as err:
        print("Error:train epoch  ---- >>", err)
        print("Error:train epoch  ---- >>", err.__traceback__.tb_lineno)
