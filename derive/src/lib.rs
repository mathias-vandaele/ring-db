use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

/// Derive macro that implements [`ringdb::Payload`] for a struct.
///
/// The storage strategy is selected with the `#[payload(storage = "...")]`
/// attribute:
///
/// | Value | Strategy | Requirement |
/// |-------|----------|-------------|
/// | `"serde"` (default) | bincode serialization, variable-size payloads | `T: Serialize + DeserializeOwned` |
/// | `"pod"` | raw bytes, zero-copy `&T` fetch | `T: bytemuck::Pod` |
///
/// # Examples
///
/// ```ignore
/// use serde::{Serialize, Deserialize};
/// use ringdb::Payload;
///
/// // Serde storage (default) — supports any serializable type
/// #[derive(Serialize, Deserialize, Payload)]
/// struct GeoRecord { lat: f64, lon: f64, label: String }
///
/// // Pod storage — zero-copy &T fetch, requires fixed-size plain-old-data
/// use bytemuck::{Pod, Zeroable};
/// #[derive(Copy, Clone, Pod, Zeroable, Payload)]
/// #[repr(C)]
/// #[payload(storage = "pod")]
/// struct GeoPoint { lat: f32, lon: f32, altitude: f32 }
/// ```
#[proc_macro_derive(Payload, attributes(payload))]
pub fn derive_payload(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Parse #[payload(storage = "pod"|"serde")]
    let mut storage = String::from("serde");
    for attr in &input.attrs {
        if attr.path().is_ident("payload") {
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("storage") {
                    let value = meta.value()?;
                    let s: syn::LitStr = value.parse()?;
                    storage = s.value();
                }
                Ok(())
            });
        }
    }

    let expanded = match storage.as_str() {
        "pod" => quote! {
            impl #impl_generics ::ringdb::Payload for #name #ty_generics #where_clause {
                type Store   = ::ringdb::__private::PodStore<Self>;
                type Builder = ::ringdb::__private::PodStoreBuilder<Self>;

                fn make_builder() -> ::ringdb::__private::Result<Self::Builder> {
                    ::ringdb::__private::PodStoreBuilder::new()
                }

                fn load_store(dir: &::std::path::Path) -> ::ringdb::__private::Result<Self::Store> {
                    ::ringdb::__private::PodStore::load(&dir.join("payloads.bin"))
                }
            }
        },
        _ => quote! {
            impl #impl_generics ::ringdb::Payload for #name #ty_generics #where_clause {
                type Store   = ::ringdb::__private::SerdeStore<Self>;
                type Builder = ::ringdb::__private::SerdeStoreBuilder<Self>;

                fn make_builder() -> ::ringdb::__private::Result<Self::Builder> {
                    ::ringdb::__private::SerdeStoreBuilder::new()
                }

                fn load_store(dir: &::std::path::Path) -> ::ringdb::__private::Result<Self::Store> {
                    ::ringdb::__private::SerdeStore::load(
                        &dir.join("payloads.bin"),
                        &dir.join("offsets.bin"),
                    )
                }
            }
        },
    };

    TokenStream::from(expanded)
}
