// 等待页面加载完成
window.addEventListener("load", () => {
    // 检查 Three.js 和 Cannon-es 是否存在
    if (!window.THREE || !window.CANNON) {
      console.error("Three.js 或 Cannon-es 未加载！");
      return;
    }
  
    // ===== Three.js 初始化 =====
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xdddddd);
  
    const canvas = document.getElementById("marblesCanvas");
    const renderer = new THREE.WebGLRenderer({ antialias: true, canvas: canvas });
    renderer.setSize(window.innerWidth, 500);
  
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / 500,
      0.1,
      1000
    );
    camera.position.set(0, 5, 15);
  
    // 添加环境光
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    // 添加方向光
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 20, 10);
    scene.add(directionalLight);
  
    // 添加 OrbitControls 控制（如果需要用户交互）
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
  
    // ===== Cannon-es 初始化 =====
    const world = new CANNON.World({
      gravity: new CANNON.Vec3(0, -9.82, 0),
    });
    world.broadphase = new CANNON.NaiveBroadphase();
    world.solver.iterations = 10;
  
    // 创建地面（Cannon-es）
    const groundBody = new CANNON.Body({
      mass: 0, // 静态地面
      shape: new CANNON.Plane(),
      material: new CANNON.Material({ friction: 0.4, restitution: 0.3 }),
    });
    groundBody.quaternion.setFromEuler(-Math.PI / 2, 0, 0);
    world.addBody(groundBody);
  
    // 创建地面（Three.js）
    const groundGeometry = new THREE.PlaneGeometry(50, 50);
    const groundMaterial = new THREE.MeshPhongMaterial({ color: 0x808080 });
    const groundMesh = new THREE.Mesh(groundGeometry, groundMaterial);
    groundMesh.rotation.x = -Math.PI / 2;
    scene.add(groundMesh);
  
    // ===== 创建弹珠 =====
    const marbles = []; // 用于存储弹珠对象
  
    // 添加弹珠的函数
    function addMarble(position) {
      // 创建 Three.js 球体
      const sphereGeometry = new THREE.SphereGeometry(0.5, 32, 32);
      const sphereMaterial = new THREE.MeshPhongMaterial({
        color: Math.random() * 0xffffff,
        shininess: 100,
      });
      const sphereMesh = new THREE.Mesh(sphereGeometry, sphereMaterial);
      sphereMesh.position.copy(position);
      scene.add(sphereMesh);
  
      // 创建 Cannon-es 刚体
      const sphereShape = new CANNON.Sphere(0.5);
      const sphereBody = new CANNON.Body({ mass: 1, shape: sphereShape });
      sphereBody.position.copy(position);
      world.addBody(sphereBody);
  
      marbles.push({ mesh: sphereMesh, body: sphereBody });
    }
  
    // 添加几个初始弹珠
    addMarble(new THREE.Vector3(0, 8, 0));
    addMarble(new THREE.Vector3(2, 10, 0));
    addMarble(new THREE.Vector3(-2, 9, 1));
  
    // ===== 动画循环 =====
    const clock = new THREE.Clock();
    function animate() {
      requestAnimationFrame(animate);
      const delta = clock.getDelta();
      world.step(1 / 60, delta, 3);
  
      // 同步 Cannon-es 刚体位置到 Three.js 网格
      marbles.forEach((obj) => {
        obj.mesh.position.copy(obj.body.position);
        obj.mesh.quaternion.copy(obj.body.quaternion);
      });
  
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  });