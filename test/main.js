async function runSum() {
    console.log(' - 00');
  if (!navigator.gpu) {
    alert("WebGPU is not supported. Try Chrome or Edge.");
    return;
  }
    console.log(' - 01');

  const inputStr = document.getElementById('input').value;
  const inputArray = inputStr.split(',').map(Number);

  
    console.log(' - 02');
  if (inputArray.length !== 8) {
    alert("Please enter exactly 8 numbers.");
    return;
  }
  
    console.log(' - 03');

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  
    console.log(' - 04');
  const shaderCode = `
  struct Buffer {
    data: array<f32>
  };

  @group(0) @binding(0) var<storage, read_write> input: Buffer;

  var<workgroup> temp: array<f32, 8>;

  @compute @workgroup_size(8)
  fn main(@builtin(local_invocation_id) id: vec3<u32>) {
    var idx = id.x;

    // Shared memory for aggregation
    
    temp[idx] = input.data[idx];

    workgroupBarrier();

    // Reduce in workgroup
    if (idx < 4) {
      temp[idx] = temp[idx * 2] + temp[idx * 2 + 1];
    }
    workgroupBarrier();

    if (idx < 2) {
      temp[idx] = temp[idx * 2] + temp[idx * 2 + 1];
    }
    workgroupBarrier();

    if (idx == 0) {
      input.data[0] = temp[0] + temp[1];
    }
  }`;

  const module = device.createShaderModule({ code: shaderCode });

  const bufferSize = inputArray.length * 4;
  const inputBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true
  });

  new Float32Array(inputBuffer.getMappedRange()).set(inputArray);
  inputBuffer.unmap();

  const resultBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }]
  });

  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: {
      module,
      entryPoint: "main"
    }
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: inputBuffer } }]
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(1);
  passEncoder.end();

  commandEncoder.copyBufferToBuffer(inputBuffer, 0, resultBuffer, 0, 4);

  const gpuCommands = commandEncoder.finish();
  device.queue.submit([gpuCommands]);

  await resultBuffer.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(resultBuffer.getMappedRange())[0];
  document.getElementById('result').textContent = result.toFixed(4);
  resultBuffer.unmap();
}