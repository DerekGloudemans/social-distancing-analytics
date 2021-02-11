import cv2
import random
import colorsys
import numpy as np
import csv
import pixel_realworld as prw

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


###---------------------------------------------------------------------------
#   Given bbox info, draws rectangle around object
 
def draw_bbox(image, bboxes, classes, show_label=True,redact = True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    
    colors = np.array([[0.2,1,0.6],
                      [0.2,1,0],
                      [0.8,0.8,0.8],
                      [0.8,0.8,0.8],
                      [0.8,0.8,0.8],
                      [0.8,0.8,0.8],
                      [0.8,0.8,0.8],
                      [0.8,0.8,0.8]])
    
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[5]
        class_ind = int(bbox[4])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.3 * (image_h + image_w) / 600)

        # redact faces
        if redact:
            image = find_blur_face(coor, image)
        
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image

def find_blur_face(coor, image):
    x_min = coor[0]
    y_min = coor[1]
    x_max = coor[2]
    y_max = y_min + (coor[3] - coor[1])//3
    
    face = image[y_min:y_max, x_min:x_max]
    
    blurred_face = anonymize_face_pixelate(face, blocks=6)
    
    image[y_min:y_max, x_min:x_max] = blurred_face
    return image
    
    
#from https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/
def anonymize_face_simple(image, factor=3.0):
	# automatically determine the size of the blurring kernel based
	# on the spatial dimensions of the input image
	(h, w) = image.shape[:2]
	kW = int(w / factor)
	kH = int(h / factor)
	# ensure the width of the kernel is odd
	if kW % 2 == 0:
		kW -= 1
	# ensure the height of the kernel is odd
	if kH % 2 == 0:
		kH -= 1
	# apply a Gaussian blur to the input image using our computed
	# kernel size
	return cv2.GaussianBlur(image, (kW, kH), 0)
#from https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/

def anonymize_face_pixelate(image, blocks=5):
	# divide the input image into NxN blocks
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, blocks + 1, dtype="int")
	ySteps = np.linspace(0, h, blocks + 1, dtype="int")
	# loop over the blocks in both the x and y direction
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			# compute the starting and ending (x, y)-coordinates
			# for the current block
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]
			# extract the ROI using NumPy array slicing, compute the
			# mean of the ROI, and then draw a rectangle with the
			# mean RGB values over the ROI in the original image
			roi = image[startY:endY, startX:endX]
			(B, G, R) = [x for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(B, G, R), -1)
	# return the pixelated blurred image
	return image

def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def bboxes_ciou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    left = np.maximum(boxes1[..., 0], boxes2[..., 0])
    up = np.maximum(boxes1[..., 1], boxes2[..., 1])
    right = np.maximum(boxes1[..., 2], boxes2[..., 2])
    down = np.maximum(boxes1[..., 3], boxes2[..., 3])

    c = (right - left) * (right - left) + (up - down) * (up - down)
    iou = bboxes_iou(boxes1, boxes2)

    ax = (boxes1[..., 0] + boxes1[..., 2]) / 2
    ay = (boxes1[..., 1] + boxes1[..., 3]) / 2
    bx = (boxes2[..., 0] + boxes2[..., 2]) / 2
    by = (boxes2[..., 1] + boxes2[..., 3]) / 2

    u = (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
    d = u/c

    aw = boxes1[..., 2] - boxes1[..., 0]
    ah = boxes1[..., 3] - boxes1[..., 1]
    bw = boxes2[..., 2] - boxes2[..., 0]
    bh = boxes2[..., 3] - boxes2[..., 1]

    ar_gt = bw/bh
    ar_pred = aw/ah

    ar_loss = 4 / (np.pi * np.pi) * (np.arctan(ar_gt) - np.arctan(ar_pred)) * (np.arctan(ar_gt) - np.arctan(ar_pred))
    alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
    ciou_term = d + alpha * ar_loss

    return iou - ciou_term

###---------------------------------------------------------------------------
#   Filters out excessively overlapping bboxes (I think)
   
def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def diounms_sort(bboxes, iou_threshold, sigma=0.3, method='nms', beta_nms=0.6):
    best_bboxes = []
    return best_bboxes

def postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE=[1,1,1]):
    # print(len(pred_bbox))

    for i, pred in enumerate(pred_bbox):
        # print(pred.shape)
        conv_shape = pred.shape
        output_size = conv_shape[1]
        conv_raw_dxdy = pred[:, :, :, :, 0:2]
        conv_raw_dwdh = pred[:, :, :, :, 2:4]
        xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
        xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        xy_grid = np.tile(tf.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
        xy_grid = xy_grid.astype(np.float)

        # print(xy_grid.shape)
        # pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
        pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
        # pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
        pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
        pred[:, :, :, :, 0:4] = tf.concat([pred_xy, pred_wh], axis=-1)
        # print(pred.shape)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    # print(np.array(pred_bbox).shape)
    pred_bbox = tf.concat(pred_bbox, axis=0)
    # print(np.array(pred_bbox).shape)
    return pred_bbox

###---------------------------------------------------------------------------
#   Filters out bboxes that are not within the image, are not scaled properly, or are invalid
#   Returns remaining bboxes
   
def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    

    valid_scale=[0, np.inf]
    
    #append batch # to end
    
    # size = pred_bbox.shape[0]
    # a = np.array([[0]] * size)
    # print(a.shape)
    #convert to numpy array
    pred_bbox = np.array(pred_bbox)
    # pred_bbox = np.append(pred_bbox, a, axis = 1)
    # print(pred_bbox.shape)
    # pred_bbox[10648:, -1] = 1
    # print(pred_bbox.shape)
    

    #separate out box dimensions nd and probability
    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    
    # image_nums = pred_bbox[:, -1]
    
    #find corners of bboxes and scale properly
    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    #mask out coordinates that are not in viewable space
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    # ensures bbox scale is positive
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    # scores = pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    
    #qualities of bounding boxm - trims out all bboxes that aren't within screen, properly scaled, too low of a score
    coors, scores, probs, classes = pred_coor[mask], scores[mask], pred_prob[mask], classes[mask] #, image_nums[mask]
    bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
    
    return bboxes, probs, classes #, image_nums


###---------------------------------------------------------------------------
#   Filters out all but people and returns their bboxes

def filter_people(bboxes, probs, classes):#, image_num):
    #list of bboxes that mark a person
    people_bboxes = []
    # people_bboxes2 = []

    # takes objects primarily identified as a person and filters out ones with relatively high chances
    # of being non-human
    for i, prob in enumerate(probs): 
        if classes[i] == 0:
            #commonly mistaken objects
            light = prob[9]
            fire = prob[10]
            stop = prob[11]
            parking = prob[12]
            bench = prob[13]
            #print(prob[9:14])
            if (light < 0.002 and fire < 0.002 and stop < 0.002 and parking < 0.002 and bench < 0.002): 
                people_bboxes.append(bboxes[i])
                # if image_num[i] == 0:
                #     people_bboxes.append(bboxes[i])
                # elif image_num[i] ==1:
                #     people_bboxes2.append(bboxes[i])

    people_bboxes = np.array(people_bboxes)
    # people_bboxes2 = np.array(people_bboxes2)

    return people_bboxes #, people_bboxes2


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)
def unfreeze_all(model, frozen=False):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            unfreeze_all(l, frozen)
            

###---------------------------------------------------------------------------
#   Outputs oocupancy data to a txt file

def video_write_info(f, real_ftpts, dt, count, people, avg_dist, avg_min_dist,cam_name,veh_pts):
    pts = real_ftpts.tolist()
    veh_pts = veh_pts.tolist()
    f.writerow([dt,cam_name, people, count, avg_dist, avg_min_dist, pts,veh_pts])
    

###---------------------------------------------------------------------------
#   Displays occupancy and compliance data in top right corner of video
        
def overlay_occupancy(img, errors, people, size):
    
    occupants = 'Occupants : {}  '.format(people)
    
    #compliance is 100 if no people are present
    if people > 0:
        comp = (1.0 - (float(errors) / float(people))) * 100
    else:
        comp = 100
    compliance = "Compliance: {}%".format(np.round(comp,1))

    #calculate size text will occupy, then adjust so overlay appears in top right corner
    x = size[1]
    box = cv2.getTextSize(occupants + compliance, cv2.FONT_HERSHEY_DUPLEX, 2, 3)
    offsetx = x//30 + box[0][0]
    cv2.putText(img, occupants + compliance, ((x - offsetx), (x//30)) , cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,0), 3)


###---------------------------------------------------------------------------
#   Returns point centered at bottom of bbox
        
def get_ftpts(bboxes):
    
    footpts = []
    
    #get ftpts for each bbox
    for i, bbox in enumerate(bboxes):
        
        #corner points of box
        coor = np.array(bbox[:4], dtype=np.int32)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])

        #add points at base of feet to list
        x = c1[0] + (c2[0] - c1[0]) // 2
        y = c2[1]
        pt = (x, y)
    
        footpts.append(pt) 
    
    #convert central foot points to numpy array            
    footpts = np.array([footpts])
    footpts = np.squeeze(np.asarray(footpts))
    
    return footpts

        
def person_bboxes(model, image_data, frame_size):
    #make bboxes
    # print(image_data.shape)
    pred_bbox = model.predict(image_data)
    pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
    all_bboxes, probs, classes = utils.postprocess_boxes(pred_bbox, frame_size, INPUT_SIZE, 0.25)#.25
    bboxes = utils.filter_people(all_bboxes, probs, classes)

    #only continue processing if there were people identified
    if len(bboxes) > 0:
        #get rid of redundant boxes
        bboxes = utils.nms(bboxes, 0.213, method='nms') #.213
    
    return bboxes



###---------------------------------------------------------------------------
#   

def compliance_count(mytree, real_pts):
    errors = 0
    for pt in real_pts:
        dist, ind = mytree.query(pt, k=2)
        closest = mytree.data[ind[1]]
        dist = prw.GPS_to_ft(pt, closest)
        if dist < 6:
            errors = errors + 1
    return errors    

###---------------------------------------------------------------------------
#   Finds average distance occupants are apart from each other
###

def find_dist(mytree, real_pts):
    size = len(real_pts)
    # middle = size//2
    # med_dists = [None] * size
    avgs = [None] * size
    for i, pt in enumerate(real_pts):
        
        dist, _ = mytree.query(pt, size)
        # med_dists[i] = dist[middle]
        
        #do this for every pt in the tree - see if there is a built in function for this
        others = dist[1:]
        avgs[i] = sum(others)/len(others)
        
    # med_med = med_dists[middle]
    # avg_med = sum(med_dists)/len(med_dists)
    avg_avg = sum(avgs)/len(avgs)

    avg_avg = avg_avg * (288200**2 + 364000**2)**(0.5)
    return avg_avg

def find_min_dist(mytree, real_pts):
    size = len(real_pts)
    # middle = size//2
    # med_dists = [None] * size
    all_mins = [None] * size
    for i, pt in enumerate(real_pts):
        
        dist, _ = mytree.query(pt, 2)
        # med_dists[i] = dist[middle]
        
        #do this for every pt in the tree - see if there is a built in function for this
        all_mins[i] = dist[1]
    # med_med = med_dists[middle]
    # avg_med = sum(med_dists)/len(med_dists)
    min_dist = min(all_mins)

    min_dist = min_dist * (288200**2 + 364000**2)**(0.5)
    return min_dist
    
