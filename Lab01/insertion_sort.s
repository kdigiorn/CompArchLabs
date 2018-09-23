			;##########################################
			;		Lab 01 skeleton
			;##########################################
			
			;		r0 --> xzr
			;		r3 --> Address of data_to_sort
			;		r4 --> Address of number of list elements before loaded into r7, later j iterator for sort
			;		r5 --> Address of head pointer
			;		r6 --> Address of tail pointer
			;		r7 --> Number of list elements
			;		r8 --> offset
			;		r9 --> temp holding value
			;		r10 --> temp holding value
			;		r11 --> (in swap) ptr to node 1
			;		r12 --> (in swap) ptr to node 2
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
loop1
			cmp		r7, r0
			beq		out1
			sub		r7, r7, #1
			bl		insert
			add		r8, r8, #4
			b		loop1
out1
			
			;#################################################################################
			;		Insertion Sort component, eventually calls swap function
			;#################################################################################
			
			mov		r8, #4		; r8 is i iterator (incr by 4)
			ldr		r7, [r4]		; reload number of list elements into r7
			lsl		r7, r7, #2	; multiply by 4 for consistent comparison with r8 iterator (i)
loop2
			cmp		r8, r7			; r7 is number of list elements * 4
			bge		out2
			mov		r4, r8			; r4 is j iterator
loop3
			cmp		r4, r0			; if j <= 0 then skip while loop
			ble		out3
			mov		r1, r4			; new iterator for going through linked list to j index
			mov		r12, r5		; set a ptr to the head of the list
loop4
			cmp		r1, r0			; compare r1 to 0
			beq		out4			; reached 0 (aka we have reached our desired index)
			sub		r1, r1, #4	; decrement new iterator r1 by 4
			ldr		r12, [r12, #8]	; r12 goes to the next node
			b		loop4
out4
			ldr		r11, [r12, #4]	; r11 is (j-1) element of array (r12's prev)
			ldr		r1, [r11]			; r1 gets value in r11 arr pos
			ldr		r2, [r12]			; r2 gets value in r12 arr pos
			cmp		r1, r2
			ble		out3				; branch if swap isn't needed
			bl		swap
			sub		r4, r4, #4	; decrement j
			b		loop3
out3
			add		r8, r8, #4		; increment i
			b		loop2
out2
			
			mov		r10, r5
			ldr		r0, [r10, #0]
			ldr		r10, [r10, #8]
			ldr		r1, [r10, #0]
			ldr		r10, [r10, #8]
			ldr		r2, [r10, #0]
			ldr		r10, [r10, #8]
			ldr		r3, [r10, #0]
			ldr		r10, [r10, #8]
			ldr		r4, [r10, #0]
			ldr		r10, [r10, #8]
			ldr		r5, [r10, #0]
			ldr		r10, [r10, #8]
			ldr		r6, [r10, #0]
			ldr		r10, [r10, #8]
			ldr		r7, [r10, #0]
			ldr		r10, [r10, #8]
			ldr		r8, [r10, #0]
			ldr		r10, [r10, #8]
			ldr		r9, [r10, #0]
			
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
			
			ldr		r9, [r11, #4]	; r9 now has r11's prev ptr
			ldr		r10, [r12, #8]	; r10 now has r12's next ptr
			str		r10, [r11, #8]	; r11's next ptr is now what r12's was
			str		r9, [r12, #4]	; r12's prev ptr is now what r11's was
			
			str		r12, [r11, #4]	; r12 becomes r11's previous node
			str		r11, [r12, #8]	; r11 becomes r12's next node
			
			mov		r15, r14
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
