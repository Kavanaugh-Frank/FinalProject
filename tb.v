`timescale 1ns/1ps

module cpu_testbench();
    // Clock and reset signals
    reg pcCLK, Clk;
    
    // Instantiate the CPU
    cpu DUT (
        .pcCLK(pcCLK),
        .Clk(Clk)
    );
    
    // Clock generation
    always begin
        #5 Clk = ~Clk;
        #7 pcCLK = ~pcCLK;
    end
    
    // Test sequence
    initial begin
        // Initialize signals
        Clk = 0;
        pcCLK = 0;
        
        // Dump waves for GTKWave or similar
        $dumpfile("cpu_testbench.vcd");
        $dumpvars(0, cpu_testbench);
        
        // Display initial register contents
        $display("Initial Register Contents:");
        $display("$t0 (R8): %d", DUT.RF.registers[8]);
        $display("$t1 (R9): %d", DUT.RF.registers[9]);
        $display("$t2 (R10): %d", DUT.RF.registers[10]);
      $display("$t3 (R11): %d", DUT.RF.registers[11]);
        
        // Run simulation for several clock cycles
        #50;
        
        // Display registers after the first instruction
      	$display("\nAfter first instruction:");
      	$display("$t0 (R8): %d", DUT.RF.registers[8]);
        $display("$t1 (R9): %d", DUT.RF.registers[9]);
        $display("$t2 (R10): %d", DUT.RF.registers[10]);
      	$display("$t3 (R11): %d", DUT.RF.registers[11]);
        
        
        // Display registers after the second instruction
        #50;
        $display("\nAfter second instruction (addition):");
      	$display("$t0 (R8): %d", DUT.RF.registers[8]);
        $display("$t1 (R9): %d", DUT.RF.registers[9]);
        $display("$t2 (R10): %d", DUT.RF.registers[10]);
      	$display("$t3 (R11): %d", DUT.RF.registers[11]);
      
      
      	// Display registers after the second instruction
        #50;
      	$display("\nAfter third instruction (addition):");
      	$display("$t0 (R8): %d", DUT.RF.registers[8]);
        $display("$t1 (R9): %d", DUT.RF.registers[9]);
        $display("$t2 (R10): %d", DUT.RF.registers[10]);
      	$display("$t3 (R11): %d", DUT.RF.registers[11]);
        
        // End simulation
        #10 $finish;
    end
    
    // Optional: Debugging display for each clock cycle
    always @(posedge Clk) begin
        $display("Current PC: %h", DUT.PCOut);
        $display("Current Instruction: %h", DUT.Instruction);
      	$display("ReadData 1 %h, Readdata2 %h", DUT.ReadData1, DUT.ReadData2);
    end
endmodule
