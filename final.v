module cpu(
    input pcCLK,
    input Clk
);

// PROGRAM COUNTER (component #1):

wire MUX4Out;   // 32 bits, output from MUX5 (component #17)
wire PCOut;     // 32 bits, input to PCAdder (component #2) and InstructionMemory (component #3)

ProgramCounter ProgramCounter (
    .PCIn(MUX4Out),     // input
    .Ckl(Clk),          // input
    .PCOut(PCOut)       // output
);

// PC ADDER (component #2):

wire PCOutPlus4;

PCAdder PCAdder (
    .PCIn(PCOut),       // input
    .PCOut(PCOutPlus4)  // output
);

// INSTRUCTION MEMEORY (component #3):

wire Instruction;

InstructionMemory InstructionMemory(
    .Address(PCOut),            // input
    .Clk(Clk),                  // input
    .Instruction(Instruction)   // output
);

// LEFT SHIFTER 2 BITS (component #4):

wire LeftShift2BitOut;

LeftShifterTwoBits LeftShifterTwoBits(
    .ValueIn(Instruction),          // input 
    .ValueOut(LeftShift2BitOut)     // output
);

endmodule

//
//
// INSTRUCTION MEMORY
//
//

module InstructionMemory(
    input [31:0] Address,         // 32-bit address
    input Clk,                    // Clock (not necessary for asynchronous reads but included for consistency)
    output reg [31:0] Instruction // 32-bit instruction output
);

    reg [7:0] memory [0:1023]; // Byte-addressable memory with 1024 bytes

    // Read operation (asynchronous)
    always @(*) begin
        Instruction = {memory[Address], memory[Address + 1], memory[Address + 2], memory[Address + 3]};
    end

    // Preload program instructions for simulation
    initial begin
        memory[0] = 8'h20; memory[1] = 8'h08; memory[2] = 8'h00; memory[3] = 8'h05; // addi $t0, $zero, 5
        memory[4] = 8'h20; memory[5] = 8'h09; memory[6] = 8'h00; memory[7] = 8'h0A; // addi $t1, $zero, 10
        memory[8] = 8'h01; memory[9] = 8'h09; memory[10] = 8'h50; memory[11] = 8'h20; // add $t2, $t0, $t1
        memory[12] = 8'hAC; memory[13] = 8'h0A; memory[14] = 8'h00; memory[15] = 8'h00; // sw $t2, 0($zero)
        memory[16] = 8'h8C; memory[17] = 8'h0B; memory[18] = 8'h00; memory[19] = 8'h00; // lw $t3, 0($zero)
    end

endmodule


//
//
// DATA MEMORY
//
//



module DataMemory(
    input [31:0] Address,       // 32-bit address
    input [31:0] WriteData,     // 32-bit data to write
    input MemRead, MemWrite,    // Control signals
    input Clk,                  // Clock
    output reg [31:0] ReadData  // 32-bit read data
);

    reg [7:0] memory [0:1023]; // Byte-addressable memory with 1024 bytes

    // Read operation (asynchronous)
    always @(*) begin
        if (MemRead) begin
            ReadData = {memory[Address], memory[Address + 1], memory[Address + 2], memory[Address + 3]};
        end
    end

    // Write operation (synchronous)
    always @(posedge Clk) begin
        if (MemWrite) begin
            memory[Address]     <= WriteData[31:24]; // MSB
            memory[Address + 1] <= WriteData[23:16];
            memory[Address + 2] <= WriteData[15:8];
            memory[Address + 3] <= WriteData[7:0];  // LSB
        end
    end

endmodule

//
//
// REGISTER FILE
//
//

module RegisterFile(
    input [4:0] ReadRegister1, ReadRegister2, WriteRegister, // 5-bit register addresses
    input [31:0] WriteData,                                // 32-bit data to write
    input RegWrite, Clk,                                   // RegWrite control signal and clock
    output [31:0] ReadData1, ReadData2                     // 32-bit data read outputs
);

    reg [31:0] registers [31:0]; // 32 registers of 32 bits each

    // Read operation
    assign ReadData1 = registers[ReadRegister1];
    assign ReadData2 = registers[ReadRegister2];

    // Write operation
    always @(posedge Clk) begin
        if (RegWrite) begin
            registers[WriteRegister] <= WriteData; // Write data to the selected register
        end
    end

endmodule



//
//
// ALU Control
//
//

module Control(
    input [5:0] opcode,       // 6-bit opcode
    input [5:0] funct,        // 6-bit function field (for R-type instructions)
    output reg ALUSrc,        // ALU Source
    output reg RegDst,        // Register Destination
    output reg MemWrite,      // Memory Write
    output reg MemRead,       // Memory Read
    output reg Beq,           // Branch Equal
    output reg Bne,           // Branch Not Equal
    output reg Jump,          // Jump signal
    output reg MemToReg,      // Memory to Register
    output reg RegWrite,      // Register Write
    output reg [2:0] ALUControl // 3-bit ALU control signal
);

    always @(*) begin
        // Default values for all control signals
        ALUSrc = 0;
        RegDst = 0;
        MemWrite = 0;
        MemRead = 0;
        Beq = 0;
        Bne = 0;
        Jump = 0;
        MemToReg = 0;
        RegWrite = 0;
        ALUControl = 3'bxxx;

        case (opcode)
            6'b000000: begin // R-type instructions
                RegDst = 1;
                ALUSrc = 0;
                RegWrite = 1;

                // Decode the funct field for ALUControl
                case (funct)
                    6'b100000: ALUControl = 3'b010; // ADD
                    6'b100010: ALUControl = 3'b110; // SUB
                    6'b100100: ALUControl = 3'b000; // AND
                    6'b100101: ALUControl = 3'b001; // OR
                    6'b101010: ALUControl = 3'b111; // SLT
                    default:   ALUControl = 3'bxxx; // Undefined
                endcase
            end
            6'b100011: begin // lw (load word)
                RegDst = 0;
                ALUSrc = 1;
                MemRead = 1;
                MemToReg = 1;
                RegWrite = 1;
                ALUControl = 3'b010; // ADD for address calculation
            end
            6'b101011: begin // sw (store word)
                ALUSrc = 1;
                MemWrite = 1;
                ALUControl = 3'b010; // ADD for address calculation
            end
            6'b000100: begin // beq (branch if equal)
                ALUSrc = 0;
                Beq = 1;
                ALUControl = 3'b110; // SUB for comparison
            end
            6'b000101: begin // bne (branch if not equal)
                ALUSrc = 0;
                Bne = 1;
                ALUControl = 3'b110; // SUB for comparison
            end
            6'b000010: begin // jump
                Jump = 1;
            end
            default: begin
                // Keep all control signals at default for undefined opcodes
            end
        endcase
    end

endmodule


//
//
// MUX and SIGN EXTENSION
//
//

module Mux32Bit2To1(
    input [31:0] a, b,    // 32-bit inputs
    input op,             // Control signal
    output [31:0] result  // 32-bit output
);

    // Select result based on control signal
    assign result = (op == 1'b0) ? a : b;

endmodule

module Mux5Bit2To1(
    input [4:0] a, b,    // 5-bit inputs
    input op,            // Control signal
    output [4:0] result  // 5-bit output
);

    // Select result based on control signal
    assign result = (op == 1'b0) ? a : b;

endmodule


module SignExtension(
    input [15:0] a,           // 16-bit input
    output [31:0] result      // 32-bit sign-extended output
);

    // Assign the sign-extended value
    assign result = {{16{a[15]}}, a}; // Repeat MSB of 'a' 16 times and concatenate with 'a'

endmodule


//
//
// ALU 32 Bit
//
//




module ALU32Bit(
    input [31:0] a, b,       // 32-bit inputs
    input cin,               // Initial carry-in
    input less,              // For SLT operation
    input [2:0] op,          // 3-bit operation code
    output [31:0] result,    // 32-bit result
    output cout,             // Final carry-out
    output set,              // MSB set signal
    output zero,             // Zero flag
    output g, p,             // Block generate and propagate
    output overflow          // Overflow flag
);

    wire [1:0] carry;        // Carry signals between 16-bit ALUs
    wire [1:0] G_block, P_block; // Generate and propagate signals for each block
    wire [1:0] setv;         // Set signals from each 16-bit ALU
    wire [1:0] overflow_temp;// Overflow flags from each 16-bit ALU

    // Instantiate two 16-bit ALUs
    ALU16Bit ALU0 (
        .a(a[15:0]), 
        .b(b[15:0]), 
        .cin(cin), 
        .less(less), 
        .op(op), 
        .result(result[15:0]), 
        .cout(carry[0]), 
        .set(setv[0]), 
        .zero(), 
        .g(G_block[0]), 
        .p(P_block[0]), 
        .overflow(overflow_temp[0])
    );

    ALU16Bit ALU1 (
        .a(a[31:16]), 
        .b(b[31:16]), 
        .cin(carry[0]), 
        .less(1'b0), 
        .op(op), 
        .result(result[31:16]), 
        .cout(carry[1]), 
        .set(setv[1]), 
        .zero(), 
        .g(G_block[1]), 
        .p(P_block[1]), 
        .overflow(overflow_temp[1])
    );

    // Assign final outputs
    assign cout = carry[1];                  // Final carry-out
    assign set = setv[1];                    // Set signal from MSB
    assign zero = (result == 32'b0);         // Zero flag
    assign g = G_block[1] | (G_block[0] & P_block[1]);
    assign p = P_block[1] & P_block[0];
    assign overflow = overflow_temp[1];     // Overflow from MSB
endmodule

module ALU16Bit(
    input [15:0] a, b,       // 16-bit inputs
    input cin,               // Initial carry-in
    input less,              // For SLT operation
    input [2:0] op,          // 3-bit operation code
    output [15:0] result,    // 16-bit result
    output cout,             // Final carry-out
    output set,              // MSB set signal
    output zero,             // Zero flag
    output g, p,             // Block generate and propagate
    output overflow          // Overflow flag
);

    wire [3:0] carry;        // Carry signals between 4-bit ALUs
    wire [3:0] G_block, P_block; // Generate and propagate signals for each block
    wire [3:0] setv;         // Set signals from each 4-bit ALU
    wire [3:0] overflow_temp;// Overflow flags from each 4-bit ALU

    // Instantiate four 4-bit ALUs
    FourBitALU ALU0 (
        .a(a[3:0]), 
        .b(b[3:0]), 
        .cin(cin), 
        .less(less), 
        .op(op), 
        .result(result[3:0]), 
        .cout(carry[0]), 
        .G(G_block[0]), 
        .P(P_block[0]), 
        .set(setv[0]), 
        .zero(), 
        .overflow(overflow_temp[0])
    );

    FourBitALU ALU1 (
        .a(a[7:4]), 
        .b(b[7:4]), 
        .cin(carry[0]), 
        .less(1'b0), 
        .op(op), 
        .result(result[7:4]), 
        .cout(carry[1]), 
        .G(G_block[1]), 
        .P(P_block[1]), 
        .set(setv[1]), 
        .zero(), 
        .overflow(overflow_temp[1])
    );

    FourBitALU ALU2 (
        .a(a[11:8]), 
        .b(b[11:8]), 
        .cin(carry[1]), 
        .less(1'b0), 
        .op(op), 
        .result(result[11:8]), 
        .cout(carry[2]), 
        .G(G_block[2]), 
        .P(P_block[2]), 
        .set(setv[2]), 
        .zero(), 
        .overflow(overflow_temp[2])
    );

    FourBitALU ALU3 (
        .a(a[15:12]), 
        .b(b[15:12]), 
        .cin(carry[2]), 
        .less(1'b0), 
        .op(op), 
        .result(result[15:12]), 
        .cout(carry[3]), 
        .G(G_block[3]), 
        .P(P_block[3]), 
        .set(setv[3]), 
        .zero(), 
        .overflow(overflow_temp[3])
    );

    // Assign final outputs
    assign cout = carry[3];                  // Final carry-out from MSB
    assign set = setv[3];                    // Set signal from MSB
    assign zero = (result == 16'b0);         // Zero flag
    assign g = G_block[3] | (G_block[2] & P_block[3]) | 
               (G_block[1] & P_block[3] & P_block[2]) | 
               (G_block[0] & P_block[3] & P_block[2] & P_block[1]);
    assign p = P_block[3] & P_block[2] & P_block[1] & P_block[0];
    assign overflow = overflow_temp[3];     // Overflow from MSB
endmodule

module FourBitALU(
    input [3:0] a, b,         // 4-bit inputs
    input [2:0] op,           // 3-bit operation code
    input cin,                // Initial carry-in
    input less,               // For set-on-less operation
    output [3:0] result,      // 4-bit result
    output cout,              // Carry-out
    output G, P,              // Block generate and propagate
    output set,               // Set signal for SLT
    output zero,              // Zero flag
    output overflow           // Overflow flag
);

    wire [3:0] C;             // Carry signals
    wire [3:0] g, p;          // Generate and propagate signals

    // Instantiate four 1-bit ALUs
    OneBitALU ALU0 (a[0], b[0], cin, less, op, result[0], C[0], g[0], p[0]);
    OneBitALU ALU1 (a[1], b[1], C[0], 1'b0, op, result[1], C[1], g[1], p[1]);
    OneBitALU ALU2 (a[2], b[2], C[1], 1'b0, op, result[2], C[2], g[2], p[2]);
    OneBitALU ALU3 (a[3], b[3], C[2], 1'b0, op, result[3], C[3], g[3], p[3]);

    // Instantiate CLA
    CLA cla (g[0], p[0], g[1], p[1], g[2], p[2], g[3], p[3], cin, C[0], C[1], C[2], C[3], G, P);

    // Instantiate OverflowDetection
    OverflowDetection ovf (C[2], C[3], overflow);

    // Assign additional outputs
    assign cout = C[3];                 // Carry-out from MSB
    assign set = result[3];             // MSB result used for SLT
    assign zero = (result == 4'b0000);  // Zero flag

endmodule

module OneBitALU(
    input a, b, cin,        // Inputs: single bits of a, b, and carry-in
    input less,             // For set-on-less operation
    input [2:0] op,         // Operation code
    output result,          // Result of the operation
    output cout,            // Carry-out bit
    output g, p             // Generate and propagate signals
);

    wire bcomp;             // Complement of b based on op[2]

    // Determine the value of bcomp
    assign bcomp = op[2] ? ~b : b;

    // Compute the sum and carry-out
    assign cout = (a & bcomp) | (cin & (a ^ bcomp));
    assign result = (op == 3'b000) ? (a & bcomp) :         // AND
                    (op == 3'b001) ? (a | bcomp) :         // OR
                    (op == 3'b010 || op == 3'b110) ? (a ^ bcomp ^ cin) : // ADD/SUB
                    (op == 3'b111) ? less :               // SLT
                    1'b0;                                 // Default

    // Generate and propagate signals
    assign g = a & bcomp;
    assign p = a | bcomp;

endmodule

module OverflowDetection(
    input c0,               // Carry-in to the MSB
    input c1,               // Carry-out from the MSB
    output V                // Overflow flag
);

    // Overflow occurs when MSB carry-in and carry-out differ
    assign V = c0 ^ c1;

endmodule

module CLA(
    input g0, p0, g1, p1, g2, p2, g3, p3, // Generate and propagate signals for 4 bits
    input cin,                            // Carry-in
    output C1, C2, C3, C4,                // Carry bits
    output G, P                           // Block generate and propagate
);

    // Compute carry signals
    assign C1 = g0 | (p0 & cin);
    assign C2 = g1 | (p1 & g0) | (p1 & p0 & cin);
    assign C3 = g2 | (p2 & g1) | (p2 & p1 & g0) | (p2 & p1 & p0 & cin);
    assign C4 = g3 | (p3 & g2) | (p3 & p2 & g1) | (p3 & p2 & p1 & g0) | (p3 & p2 & p1 & p0 & cin);

    // Block generate and propagate
    assign G = g3 | (p3 & g2) | (p3 & p2 & g1) | (p3 & p2 & p1 & g0);
    assign P = p3 & p2 & p1 & p0;

endmodule

//
//
// HELPERS
//
//


module ProgramCounter(PCIn, Clk, PCOut);
  input [31:0] PCIn; // 32-bit input address
  input Clk; // Clock signal
  output reg [31:0] PCOut; // 32-bit output address
  
  always @(posedge Clk) 
    begin
        PCOut <= PCIn; 
    end
endmodule

module PCAdder(PCIn, PCOut);
  input [31:0] PCIn; // 32-bit input address
  output [31:0] PCOut; // 32-bit output incremented address
  
  assign PCOut = PCIn + 32'd4; // Add 4 to PCIn and assign to PCOut
endmodule

module LeftShifterTwoBits(ValueIn, ValueOut);
  input [25:0] ValueIn; // 26-bit input 
  output [27:0] ValueOut; // 28-bit output
  
  // Shift the input left by 2 bits and append two zeros
  assign ValueOut = {ValueIn, 2'b00}; 
endmodule

module LeftShifterWithDiscard(ValueIn, ValueOut);
  input [31:0] ValueIn;   // 32-bit input
  output [31:0] ValueOut; // 32-bit output
  
  // Get the lower 30 bits, shift left by 2, and append two zeroes
  assign ValueOut = {ValueIn[29:0], 2'b00}; 
endmodule
  
module BEQAdder(ValueIn1, ValueIn2, ValueOut); 
  input [31:0] ValueIn1, ValueIn2; // 32-bit inputs
  output [31:0] ValueOut; // sum of inputs
  
  assign ValueOut = ValueIn1 + ValueIn2; // Add the two inputs
endmodule
