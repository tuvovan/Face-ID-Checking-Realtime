from keras import backend as K
K.set_image_data_format('channels_first')
from fr_utils_multiple import *
from inception_blocks_v2 import *
import openface
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[0],y_pred[1])), axis = None)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[0],y_pred[2])), axis = None)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss,0.0))
    
    return loss

# FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)
import numpy as np
import cv2
database = {}
database = np.load("database_2.npy").item()
def verify(image, identity, database, model):
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image,model)

    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding-database[identity])

    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist<0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return dist, door_open
#
# verify("tu2_0.jpg", "Tu Vo Van", database, FRmodel)

def who_is_it(img1, image, database, model, x, y):
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)

    encoding= img_to_encoding_video(img1,model)

    ## Step 2: Find the closest encoding ##
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding-db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist<min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
        cv2.putText(image, "???", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, [255,0,0], 2)
        # cv2.imshow('who??', image)
        # cv2.waitKey(3000)
        # cv2.destroyAllWindows()
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        cv2.putText(image, str(identity), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, [255,0,0], 2)
        # cv2.imshow('who??', image)
        # cv2.waitKey(3000)
        # cv2.destroyAllWindows()
    return min_dist, identity

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, image = cap.read()
    # image = cv2.imread("ah.jpg")
    if cv2.waitKey(1) & 0xFF == ord('y'):  # save on pressing 'y'
        cv2.destroyAllWindows()
        predictor_model = "shape_predictor_68_face_landmarks.dat"
        face_aligner = openface.AlignDlib(predictor_model)
        detected_faces = get_face(image)
        for face_rect in detected_faces:
            cv2.rectangle(image, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (255, 0, 0),2)
            alignedFace = face_aligner.align(96, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            who_is_it(alignedFace, image, database, FRmodel, face_rect.left(), face_rect.top())
        cv2.imshow("Recognize", image)
        cv2.waitKey(0)
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
        cv2.putText(image, "Show your Face and Press Y", (120,120), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], 2)
        cv2.imshow("Recognize", image)

cap.release()
cv2.destroyAllWindows()
