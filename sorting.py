def bubble_sort(nums):
    swapping = True
    end = len(nums)
    
    while swapping:
        swapping = False
        
        for i in range(1, end):
            if nums[i-1] > nums[i]:
                nums[i-1], nums[i] = nums[i], nums[i-1]
                swapping = True
        
    return nums

def merge_sort(nums):
    if len(nums) < 2:
        return nums
    
    first = merge_sort(nums[:len(nums) // 2])
    second = merge_sort(nums[len(nums) // 2:])
    return merge(first, second)

def merge(first, second):
    final = []
    i, j = 0, 0

    while i < len(first) and j < len(second):
        if first[i] < second[j]:
            final.append(first[i])
            i += 1
        else:
            final.append(second[j])
            j += 1
        
    while i < len(first):
        final.append(first[i])
        i += 1
        
    while j < len(second):
        final.append(second[j])
        j += 1
        
    return final

def insertion_sort(nums):
    for i in range(1, len(nums)):
       j = i
       while j > 0 and nums[j-1] > nums[j]:
           nums[j-1], nums[j] = nums[j], nums[j-1]
           j -= 1
           
    return nums

def quick_sort(nums, low, high):
	if low < high:
		p = partition(nums, low, high)
		quick_sort(nums, low, p - 1)
		quick_sort(nums, p + 1, high)
		
def partition(nums, low, high):
	pivot = nums[high]
	i = low
	for j in range(low, high):
		if nums[j] < pivot:
			nums[i], nums[j] = nums[j], nums[i]
			i += 1
	nums[i], nums[high] = nums[high], nums[i]
	return i

def selection_sort(nums):
    for i in range(0, len(nums)):
        smallest_idx = i
        for j in range(i + 1, len(nums)):
            if nums[j] < nums[smallest_idx]:
                smallest_idx = j
        nums[i], nums[smallest_idx] = nums[smallest_idx], nums[i]

    return nums
