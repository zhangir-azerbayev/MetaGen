#The files are output from Detectron2. They contian the COCO coding of the object categories

#files is a list of files to read
function read_output_from_detectron(files)
    #get the percepts in number form. Each percept/video is a file, each line is a frame
    gt_percepts = []
    n_percepts = length(files)
    n_frames = 10
    #for each file
    for p = 1:n_percepts

    	#open input file
        #input_file = open(files[p])
        #slurp = read(input_file, String)

        open(files[p]) do file2
        	#percept is a matrix, just like in the GM
    	    percept = []
    	    f = 1 #f keeps track of the frame we're on

            #each percept should have the number of frames specified in the GM
            @assert countlines(files[p]) == n_frames

            #for each frame
            for line in enumerate(eachline(file2))

                #line is a tuple consisting of the line number and the string
                #changing to just the string
                line=line[2]

          		start = findfirst("[",line)[1]
          		finish = findfirst("]",line)[1]
          		pred = line[start+1:finish-1]
          		arr = split(pred, ", ")
          		frame = Array{Int}(undef, length(arr))
          		for j = 1:length(arr)
          		    #add 1 to fix indexing discrepancy between python indices for COCO dataset and Julia indexing
          		    frame[j] = parse(Int,arr[j])+1
          		end

          		#remove objects not in possible_objects, aka find overlap between set of possible_objects and g
    			frame_real = intersect(frame, names_to_IDs(possible_objects, possible_objects)) #problem: intersect isn't returning duplicates

                #add the frame to the percept
                push!(percept, possible_objects[frame_real])

    			f = f + 1
            end

            println("percept ", percept)

            push!(gt_percepts,percept)
        end
    end
    return gt_percepts
end
