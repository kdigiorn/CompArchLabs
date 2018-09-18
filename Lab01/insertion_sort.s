			;##########################################
			;		Lab 01 skeleton
			;##########################################
			
			;		r3 --> Address of data_to_sort
			;		r4 --> Address of number of list elements
			;		r5 --> Address of head pointer
			;		r6 --> Address of tail pointer
			;		r7 --> Number of list elements
			;		r8 --> offset
			;		r9 --> temp holding value
			;		out --> Label for outside the loop
			
data_to_sort	dcd		34, 23, 22, 8, 50, 74, 2, 1, 17, 40
list_elements	dcd		10
			
main
			ldr		r3, =data_to_sort   ; Load the starting address of the first
			;		of element of the array of numbers into r3
			ldr		r4, =list_elements  ; address of number of elements in the list
			
			add		r5, r3, #400	    ; location of first element of linked list - "head pointer"
			;		(note that I assume it is at a "random" location
			;		beyond the end of the array.)
			
			
			;#################################################################################
			;		Include any setup code here prior to loop that loads data elements in array
			;#################################################################################
			
			ldr		r7, [r4]	; load address of number of elements
			mov		r6, r5
			
			;#################################################################################
			;		Start a loop here to load elements in the array and add them to a linked list
			;#################################################################################
			
			mov		r0, #0
			mov		r8, #0
loop
			cmp		r7, r0
			beq		out
			sub		r7, r7, #1
			bl		insert
			add		r8, r8, #4
			b		loop
out


			end
			;#################################################################################
			;		Add insert, swap, delete functions
			;#################################################################################
			
insert
			ldr		r9, [r3, r8] 	; Loading the value into a temp
			str		r9, [r6]			; Storing r9 to tail pointer
			sub		r9, r6, #32		; Getting the address of the prev
			str		r9, [r6, #4]
			add		r9, r6, #32		; Address of next
			str		r9, [r6, #8]
			add		r6, r6, #32
			mov		r15, r14
			
swap
			ldr		r9, [r11, #4]	; load prev from r11
			add		r9, r9, #8		; increment r9 ptr to contain next addr
			str		r12, [r9]			; store value of r12 into address given by r9
			ldr		r9, [r12, #8]	; load next from r12
			add		r9, r9, #4		; increment r9 ptr to contain prev addr
			str		r11, [r9]			; store value of r11 into address given by r9
			ldr		r9, [r11, #4]	; r11 is the addr of node1 to be swapped
			ldr		r10, [r12, #4]	; r12 is addr of node2 to be swapped. Values of prev ptrs in r9 & 10
			str		r10, [r11, #4]	; swap prev ptrs
			str		r9, [r12, #4]
			ldr		r9, [r11, #8]	; same process with next ptrs
			ldr		r10, [r12, #8]
			str		r10, [r11, #8]
			str		r9, [r12, #8]
			mov		r15, r14














			
			
