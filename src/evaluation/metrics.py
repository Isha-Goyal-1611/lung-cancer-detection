def calculate_iou(box1,box2):
    """box format:(x_center,y_center,width,height)"""
    x_center1,y_center1,width1,height1=box1
    x_center2,y_center2,width2,height2=box2

    x_min1= x_center1-width1/2
    x_max1= x_center1+width1/2
    y_min1= y_center1-height1/2
    y_max1= y_center1+height1/2

    x_min2= x_center2-width2/2
    x_max2= x_center2+width2/2
    y_min2= y_center2-height2/2
    y_max2= y_center2+height2/2

    intersection_x_min=max(x_min1,x_min2)
    intersection_y_min=max(y_min1,y_min2)
    intersection_x_max=min(x_max1,x_max2)
    intersection_y_max=min(y_max1,y_max2)

    if intersection_x_max<intersection_x_min or intersection_y_max<intersection_y_min:
        return 0.0
    else:
        intersection_area=(intersection_x_max-intersection_x_min)*(intersection_y_max-intersection_y_min)
        area1=width1*height1
        area2=width2*height2
        union=area1+area2-intersection_area
        return intersection_area/union
    pass

if __name__=='__main__':
    box1=(50,50,20,20)
    box2=(55,55,20,20)
    box3=(200,200,20,20)
    print("IoU between box1 and box2:",calculate_iou(box1,box2))
    print("IoU between box1 and box3:",calculate_iou(box1,box3))
