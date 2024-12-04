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
        
        // Run additional simulation time (for example, 1600ns)
        #3000;
        
        // Display selected registers after 100ns of simulation
        display_selected_registers();
        
        // End simulation
        #10 $finish;
    end
    
    // Task to display selected registers (s0, s1, s2, t0, t1, t2, t3, t4)
    task display_selected_registers;
        begin
            // Display $s0, $s1, $s2, and $t0-$t4
            $display("Register $s0 (R16): %d", DUT.RF.registers[16]);
            $display("Register $s1 (R17): %d", DUT.RF.registers[17]);
            $display("Register $s2 (R18): %d", DUT.RF.registers[18]);
            $display("Register $t0 (R8): %d", DUT.RF.registers[8]);
            $display("Register $t1 (R9): %d", DUT.RF.registers[9]);
            $display("Register $t2 (R10): %d", DUT.RF.registers[10]);
            $display("Register $t3 (R11): %d", DUT.RF.registers[11]);
            $display("Register $t4 (R12): %d", DUT.RF.registers[12]);
        end
    endtask
endmodule
