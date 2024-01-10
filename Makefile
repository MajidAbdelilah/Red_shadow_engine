SRCS =	 ExampleBase/ExampleBase.c Texturing/Texturing.c
OBJS = $(SRCS:.c=.o)
SRCS_SSE =	 math.sse.c
OBJS_SEE = $(SRCS_SSE:.sse.c=.sse.o)
BONUS_SRCS =
BONUS_OBJS = $(BONUS_SRCS:.c=.o)
GNL_SRCS =
GNL_OBJS = $(GNL_SRCS:.c=.o)
CC = gcc
AR = ar
C_FLAGS = -Wall -Ofast -g -I./ExampleBase -I./# -O3 -g -fsanitize=address
LD_FLAGS := -lm -L./ -l:libLLGL.so
NAME = fdf
LD_FLAGS_OBJ = 
.PHONY: clean

all: $(NAME)


clean:
	rm -rf $(OBJS)  $(BONUS_OBJS) $(GNL_OBJS) debug bonus b_debug main.o $(OBJS_SEE)

fclean: clean
	rm -rf $(NAME)

re: fclean all

%.sse.o: %.sse.c
	$(CC) $(C_FLAGS) -c -o $@ $< $(LD_FLAGS_OBJ)

%.o: %.c
	$(CC) $(C_FLAGS) -c -o $@ $< $(LD_FLAGS_OBJ)

$(NAME): $(OBJS)
	$(CC) $(C_FLAGS) -o  $@ $(OBJS) $(LD_FLAGS) 

debug: C_FLAGS+= -g
debug: LD_FLAGS+= -fsanitize=address
debug: LD_FLAGS_OBJ+= -fsanitize=address
debug:  $(OBJS)
	$(CC) $(C_FLAGS) -o $@ $(OBJS)  $(LD_FLAGS)
