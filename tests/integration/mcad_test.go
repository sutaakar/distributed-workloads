/*
Copyright 2023.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package integration

import (
	"bytes"
	"html/template"
	"os/exec"
	"strings"
	"testing"

	. "github.com/onsi/gomega"
	support "github.com/project-codeflare/codeflare-operator/test/support"
	mcadv1beta1 "github.com/project-codeflare/multi-cluster-app-dispatcher/pkg/apis/controller/v1beta1"
	rayv1alpha1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1alpha1"

	authenticationv1 "k8s.io/api/authentication/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/yaml"
)

var notebookResource = schema.GroupVersionResource{Group: "kubeflow.org", Version: "v1", Resource: "notebooks"}

type NotebookProps struct {
	IngressDomain           string
	OpenShiftApiUrl         string
	KubernetesBearerToken   string
	Namespace               string
	OpenDataHubNamespace    string
	CodeFlareImageStreamTag string
	JobType                 string
	NotebookPVC             string
}

func TestMnistPyTorchMCAD(t *testing.T) {
	test := support.With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	// Test configuration
	config := &corev1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "notebooks-mcad",
		},
		BinaryData: map[string][]byte{
			// MNIST MCAD Notebook
			"mnist_mcad_mini.ipynb": ReadFile(test, "resources/mnist_mcad_mini.ipynb"),
		},
		Immutable: support.Ptr(true),
	}
	config, err := test.Client().Core().CoreV1().ConfigMaps(namespace.Name).Create(test.Ctx(), config, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created ConfigMap %s/%s successfully", config.Namespace, config.Name)

	// Create PVC for Notebook
	notebookPVC := &corev1.PersistentVolumeClaim{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "PersistentVolumeClaim",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "jupyterhub-nb-kube-3aadmin-pvc",
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{
				corev1.ReadWriteOnce,
			},
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: resource.MustParse("10Gi"),
				},
			},
		},
	}
	notebookPVC, err = test.Client().Core().CoreV1().PersistentVolumeClaims(namespace.Name).Create(test.Ctx(), notebookPVC, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created PersistentVolumeClaim %s/%s successfully", notebookPVC.Namespace, notebookPVC.Name)

	// Read the Notebook CR from resources and perform replacements for custom values using go template
	token := createTestMnistPyTorchMCADRBAC(test, namespace)
	notebookProps := NotebookProps{
		IngressDomain:           getIngressDomain(test),
		OpenShiftApiUrl:         getOpenShiftApiUrl(test),
		KubernetesBearerToken:   token,
		Namespace:               namespace.Name,
		OpenDataHubNamespace:    GetOpenDataHubNamespace(),
		CodeFlareImageStreamTag: getCodeFlareImageStreamTag(test),
		JobType:                 "mcad",
		NotebookPVC:             "jupyterhub-nb-kube-3aadmin-pvc",
	}
	notebookTemplate := string(ReadFile(test, "resources/custom-nb-small.yaml"))
	parsedNotebookTemplate, err := template.New("notebook").Parse(notebookTemplate)
	test.Expect(err).NotTo(HaveOccurred())

	// Filter template and store results to the buffer
	notebookBuffer := new(bytes.Buffer)
	err = parsedNotebookTemplate.Execute(notebookBuffer, notebookProps)
	test.Expect(err).NotTo(HaveOccurred())

	// Create Notebook CR
	notebookCR := &unstructured.Unstructured{}
	err = yaml.NewYAMLOrJSONDecoder(notebookBuffer, 8192).Decode(notebookCR)
	test.Expect(err).NotTo(HaveOccurred())
	_, err = test.Client().Dynamic().Resource(notebookResource).Namespace(namespace.Name).Create(test.Ctx(), notebookCR, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())

	// Make sure the AppWrapper is created and running
	test.Eventually(support.AppWrappers(test, namespace), support.TestTimeoutLong).
		Should(
			And(
				HaveLen(1),
				ContainElement(WithTransform(support.AppWrapperName, HavePrefix("mnistjob"))),
				ContainElement(WithTransform(support.AppWrapperState, Equal(mcadv1beta1.AppWrapperStateActive))),
			),
		)

	// Make sure the AppWrapper finishes and is deleted
	test.Eventually(support.AppWrappers(test, namespace), support.TestTimeoutLong).
		Should(HaveLen(0))
}

func TestMCADRay(t *testing.T) {
	test := support.With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	// Test configuration
	config := &corev1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "notebooks-ray",
		},
		BinaryData: map[string][]byte{
			// MNIST MCAD Notebook
			"mnist_ray_mini.ipynb": ReadFile(test, "resources/mnist_ray_mini.ipynb"),
			"mnist.py":             ReadFile(test, "resources/mnist.py"),
			"requirements.txt":     ReadFile(test, "resources/requirements.txt"),
		},
		Immutable: support.Ptr(true),
	}
	config, err := test.Client().Core().CoreV1().ConfigMaps(namespace.Name).Create(test.Ctx(), config, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created ConfigMap %s/%s successfully", config.Namespace, config.Name)

	// Create PVC for Notebook
	notebookPVC := &corev1.PersistentVolumeClaim{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "PersistentVolumeClaim",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "jupyterhub-nb-kube-3aadmin-pvc",
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{
				corev1.ReadWriteOnce,
			},
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: resource.MustParse("10Gi"),
				},
			},
		},
	}
	notebookPVC, err = test.Client().Core().CoreV1().PersistentVolumeClaims(namespace.Name).Create(test.Ctx(), notebookPVC, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created PersistentVolumeClaim %s/%s successfully", notebookPVC.Namespace, notebookPVC.Name)

	// Read the Notebook CR from resources and perform replacements for custom values using go template
	token := createTestMnistPyTorchMCADRBAC(test, namespace)
	notebookProps := NotebookProps{
		IngressDomain:           getIngressDomain(test),
		OpenShiftApiUrl:         getOpenShiftApiUrl(test),
		KubernetesBearerToken:   token,
		Namespace:               namespace.Name,
		OpenDataHubNamespace:    GetOpenDataHubNamespace(),
		CodeFlareImageStreamTag: getCodeFlareImageStreamTag(test),
		JobType:                 "ray",
		NotebookPVC:             "jupyterhub-nb-kube-3aadmin-pvc",
	}
	notebookTemplate := string(ReadFile(test, "resources/custom-nb-small.yaml"))
	parsedNotebookTemplate, err := template.New("notebook").Parse(notebookTemplate)
	test.Expect(err).NotTo(HaveOccurred())

	// Filter template and store results to the buffer
	notebookBuffer := new(bytes.Buffer)
	err = parsedNotebookTemplate.Execute(notebookBuffer, notebookProps)
	test.Expect(err).NotTo(HaveOccurred())

	// Create Notebook CR
	notebookCR := &unstructured.Unstructured{}
	err = yaml.NewYAMLOrJSONDecoder(notebookBuffer, 8192).Decode(notebookCR)
	test.Expect(err).NotTo(HaveOccurred())
	_, err = test.Client().Dynamic().Resource(notebookResource).Namespace(namespace.Name).Create(test.Ctx(), notebookCR, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())

	// Make sure the AppWrapper is created and running
	test.Eventually(support.AppWrappers(test, namespace), support.TestTimeoutLong).
		Should(
			And(
				HaveLen(1),
				ContainElement(WithTransform(support.AppWrapperName, HavePrefix("mnistjob"))),
				ContainElement(WithTransform(support.AppWrapperState, Equal(mcadv1beta1.AppWrapperStateActive))),
			),
		)

	// Make sure the AppWrapper finishes and is deleted
	test.Eventually(support.AppWrappers(test, namespace), support.TestTimeoutLong).
		Should(HaveLen(0))
}

func getIngressDomain(test support.Test) string {
	domain, err := executeCommand("oc", "get", "ingresses.config/cluster", "-o", "jsonpath={.spec.domain}")
	test.T().Logf("Domain %s", domain)
	test.Expect(err).NotTo(HaveOccurred())
	return domain
}

func getOpenShiftApiUrl(test support.Test) string {
	openShiftApiUrl, err := executeCommand("oc", "whoami", "--show-server=true")
	openShiftApiDomain := strings.TrimPrefix(openShiftApiUrl, "https://")
	test.T().Logf("Domain %s", openShiftApiDomain)
	test.Expect(err).NotTo(HaveOccurred())
	return openShiftApiDomain
}

func executeCommand(name string, arg ...string) (string, error) {
	outputBytes, err := exec.Command(name, arg...).CombinedOutput()
	return string(outputBytes), err
}

func getCodeFlareImageStreamTag(test support.Test) string {
	cfis, err := test.Client().Image().ImageV1().ImageStreams(GetOpenDataHubNamespace()).Get(test.Ctx(), "codeflare-notebook", metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.Expect(cfis.Spec.Tags).To(HaveLen(1))
	return cfis.Spec.Tags[0].Name
}

func createTestMnistPyTorchMCADRBAC(test support.Test, namespace *corev1.Namespace) (token string) {
	serviceAccount := &corev1.ServiceAccount{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "ServiceAccount",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mcad-test-user",
			Namespace: namespace.Name,
		},
	}
	serviceAccount, err := test.Client().Core().CoreV1().ServiceAccounts(namespace.Name).Create(test.Ctx(), serviceAccount, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())

	role := &rbacv1.Role{
		TypeMeta: metav1.TypeMeta{
			APIVersion: rbacv1.SchemeGroupVersion.String(),
			Kind:       "Role",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mcad-test-role",
			Namespace: namespace.Name,
		},
		Rules: []rbacv1.PolicyRule{
			{
				Verbs: []string{"get", "create", "delete", "list", "patch", "update"},
				// APIGroups: []string{mcadv1beta1.GroupName},
				APIGroups: []string{"mcad.ibm.com"},
				Resources: []string{"appwrappers"},
			},
			{
				Verbs:     []string{"get", "list"},
				APIGroups: []string{rayv1alpha1.GroupVersion.Group},
				Resources: []string{"rayclusters", "rayclusters/status"},
			},
			{
				Verbs:     []string{"get", "list"},
				APIGroups: []string{"route.openshift.io"},
				Resources: []string{"routes"},
			},
		},
	}
	role, err = test.Client().Core().RbacV1().Roles(namespace.Name).Create(test.Ctx(), role, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())

	roleBinding := &rbacv1.RoleBinding{
		TypeMeta: metav1.TypeMeta{
			APIVersion: rbacv1.SchemeGroupVersion.String(),
			Kind:       "RoleBinding",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "mcad-test",
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: rbacv1.SchemeGroupVersion.Group,
			Kind:     "Role",
			Name:     role.Name,
		},
		Subjects: []rbacv1.Subject{
			{
				Kind:      "ServiceAccount",
				APIGroup:  corev1.SchemeGroupVersion.Group,
				Name:      serviceAccount.Name,
				Namespace: serviceAccount.Namespace,
			},
		},
	}
	_, err = test.Client().Core().RbacV1().RoleBindings(namespace.Name).Create(test.Ctx(), roleBinding, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())

	treq := &authenticationv1.TokenRequest{
		Spec: authenticationv1.TokenRequestSpec{
			ExpirationSeconds: support.Ptr(int64(3600)),
		},
	}
	treq, err = test.Client().Core().CoreV1().ServiceAccounts(namespace.Name).CreateToken(test.Ctx(), "mcad-test-user", treq, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	return treq.Status.Token
}
