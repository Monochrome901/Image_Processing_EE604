
  # STEPS FOR LOW_LIGHT PHOTOGRAPHY


- convert from bgr to rgb  
- convert from int to double and normalize them (division by 255)  
- apply shadow mask to preserve shadows 
  
  
- ## shadow mask algo -  
	 1. find luminescence (brightness) of the image  
	2. diffence taken to find spots which are significantly brighter  
	3. flag is used to mark spots in a binary matrix (0-1 matrix), where shadows might be present  
	4. apply flood fill, erode and dialation (morphological functions) to remove noise from mask  
	5. apply gausian filter to smoothen the edges of the mask  
return mask  
  
 
- flash and no flash images are split into color components  
- bilateral filter is computed for both flash and no flash for r,g,b components separately (implemented from scratch)  
  
  
- ## bilateral filter algo -  
	1. define parameters- standard deviations and kernel size  
(s_s - spatial effect, assigns weights to distances), (s_r - intensity effect, assigns weight to intensity difference, smaller it is, more the edges are preserved), (ws, kernel size)  
	2. define a gaussian filter of size ws and std - s_s and pad the images with half the size of filter  
	3. define three outputs that we require jbf(joint bilateral filter), flash_base and no_flash_base  
	4. interation over the image - (below tasks are done for both flash and no_flash images)  
		- crop out a square of size of the gaussain filter  
		- subtract it from the centre pixel, helps in simplyfying code  
		- find the intensity mask for the cutout (this helps in assigning wieghts for intensities, use standard deviation of s_r)  
		- make a mask for both no_flash and flash by combining gaussain with intensity weights(normalize it)  
		- take the mask from flash and apply it on the no_flash cutout to create a joint no_flash mask (flash image contains a much better estimate of the true high-frequency(edges) information than the ambient image)  
		- now apply flash mask to flash cutout(gives flash_base), do the same for no_flash  
		- sum the matrix up and add give the value of the centre pixel  
	5. return joint_no_flash, flash_base, no_flash_base  
 
 - stack the individual color components for all three of the above and make joint_image, flash_base, and ambient_base
 - dividing flash image by flash_base image, gives us the details or the high frequency parts, do it get flash_details
 - ## SUMMARY OF PRODUCTS
	  1. shadow_mask
	  2. flash_details
	  3. no_flash_base or ambient_base
	  4.  joint_image
- # COMBINE THIS IN A FORMULA
	`final_image = (1 - shadow_mask)*joint_image*flash_details + shadow_details*ambient_base`
	
- we still do require shadow filtering because flash shadows (shadows because of flashes) might be given less weights
- flash details is used to bring the details which are lost in no_flash_image
