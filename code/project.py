import cv2 as cv
import numpy as np
import os

# Helper function to calculate angle cosine
def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

# Finds squares (males) in the image
def findSquares(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (5, 5), 0)
    all_squares = []
    
    for thrs in range(0, 255, 26):
        if thrs == 0:
            bin_img = cv.Canny(img_gray, 0, 50, apertureSize=5)
            bin_img = cv.dilate(bin_img, None)
        else:
            ret, bin_img = cv.threshold(img_gray, thrs, 255, cv.THRESH_BINARY)
        
        contours, hierarchy = cv.findContours(bin_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            cnt_len = cv.arcLength(cnt, True)
            cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)
            if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
                cnt = cnt.reshape(-1, 2)
                max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                (x, y, w, h) = cv.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if max_cos < 0.15 and 0.90 <= aspect_ratio <= 1.10:
                    all_squares.append(cnt)

    unique_squares = []
    centroids = []
    distance_threshold = 20
    for sq in all_squares:
        M = cv.moments(sq)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            is_duplicate = any(np.sqrt((cx - ucx)**2 + (cy - ucy)**2) < distance_threshold for ucx, ucy in centroids)
            if not is_duplicate:
                unique_squares.append(sq)
                centroids.append((cx, cy))
    return unique_squares

# Finds circles (females) in the image
def findCircles(gray_img):
    circles = cv.HoughCircles(gray_img, cv.HOUGH_GRADIENT, dp=1, 
                             minDist=40,
                             param1=50, 
                             param2=30, 
                             minRadius=15, 
                             maxRadius=30) 
    print(circles if circles is not None else "no females")
    return circles[0] if circles is not None else []

# Writes simplified 3-column PED file
def generate_ped_file(individuals_list, filename="output.ped"):
    print(f"\n--- Generating {filename} ---")
    
    try:
        with open(filename, 'w') as f:
            f.write("#IND_ID\tSEX\tPHENOTYPE\n")
            
            for ind in individuals_list:
                ind_id = str(ind['id'])
                sex = "1" if ind['type'] == 'male' else "2"
                phenotype = "2" if ind['is_affected'] else "1"
                
                f.write(f"{ind_id}\t{sex}\t{phenotype}\n")
                
        print(f"Successfully created 3-column file: {filename}")
    
    except PermissionError:
        print(f"\nERROR: Permission denied.")
        print(f"Could not write to the file: {filename}")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred while writing the file:")
        print(e)


# Main Analysis Function
def analyze_pedigree(image_path):
    counts = {'affected_males': 0, 'unaffected_males': 0, 'total_males': 0,
              'affected_females': 0, 'unaffected_females': 0, 'total_females': 0}
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None

    output_image = image.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    temp_individuals = [] 
    males = findSquares(image)
    females = findCircles(gray)
    
    SHADING_THRESHOLD = 128 

    for sq in males:
        counts['total_males'] += 1
        mask = np.zeros(gray.shape, dtype="uint8")
        cv.drawContours(mask, [sq], -1, 255, -1)
        mean_val = cv.mean(gray, mask=mask)[0]
        is_affected = mean_val < SHADING_THRESHOLD
        
        (x, y, w, h) = cv.boundingRect(np.array(sq))
        M = cv.moments(sq)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        if is_affected: counts['affected_males'] += 1
        else: counts['unaffected_males'] += 1
        
        temp_individuals.append({
            'type': 'male', 'centroid': (cx, cy), 'bbox': (x, y, w, h), 
            'is_affected': is_affected
        })
        
    for f in females:
        counts['total_females'] += 1
        x, y, r = int(f[0]), int(f[1]), int(f[2])
        mask = np.zeros(gray.shape, dtype="uint8")
        cv.circle(mask, (x, y), r, 255, -1)
        mean_val = cv.mean(gray, mask=mask)[0]
        is_affected = mean_val < SHADING_THRESHOLD

        if is_affected: counts['affected_females'] += 1
        else: counts['unaffected_females'] += 1

        temp_individuals.append({
            'type': 'female', 'centroid': (x, y), 'radius': r, 
            'is_affected': is_affected
        })

    if not temp_individuals:
        print("No individuals found.")
        return counts, output_image, [] 

    # 'Generation Bucketing' Sorting Logic
    y_coords = sorted(list(set([ind['centroid'][1] for ind in temp_individuals])))
    generation_levels = []
    gen_y_level = y_coords[0]
    generation_levels.append(gen_y_level)
    Y_GENERATION_THRESHOLD = 30 
    
    for y in y_coords:
        if y - gen_y_level > Y_GENERATION_THRESHOLD:
            gen_y_level = y
            generation_levels.append(gen_y_level)

    for ind in temp_individuals:
        closest_gen_y = min(generation_levels, key=lambda y: abs(y - ind['centroid'][1]))
        ind['generation'] = generation_levels.index(closest_gen_y)

    sorted_individuals = sorted(temp_individuals, key=lambda i: (i['generation'], i['centroid'][0]))
    # End of Sorting Logic

    final_individuals_list = []
    for i, ind in enumerate(sorted_individuals):
        ind_id = i + 1 
        ind['id'] = ind_id
        final_individuals_list.append(ind)
        
        label = f"{ind_id}"
        
        if ind['type'] == 'male':
            (x, y, w, h) = ind['bbox']
            color = (0, 0, 255) if ind['is_affected'] else (0, 150, 0)
            cv.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
            cv.putText(output_image, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            (x, y) = ind['centroid']
            r = ind['radius']
            color = (0, 0, 255) if ind['is_affected'] else (0, 150, 0)
            cv.circle(output_image, (x, y), r, color, 2)
            cv.putText(output_image, label, (x - r, y - r - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return counts, output_image, final_individuals_list

# Main Execution

import matplotlib.pyplot as plt
im_path = './sample.png' 
img = cv.imread(im_path)
plt.imshow(img)
plt.show()
analysis_results, visual_output, individuals_list = analyze_pedigree(im_path)

if analysis_results:
    print("--- Pedigree Shape Analysis Results ---")
    for key, value in analysis_results.items():
        print(f"- {key.replace('_', ' ').title()}: {value}")

    if individuals_list:
        output_dir = "/home/work/College/FCV"
        ped_file_path = os.path.join(output_dir, "output.ped")
        
        print(f"\nAttempting to save PED file to: {ped_file_path}")
        generate_ped_file(individuals_list, ped_file_path)
    else:
        print("No individuals found, cannot generate PED file.")
    
    cv.imshow("Final Output (Generation-Sorted)", visual_output)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Analysis failed. Check image path.")
